import numpy as np
import os
import torch
from .giant_impact_phase_emulator import get_sim_copy, align_simulation, get_rad, perfect_merge

#calculate log(1 + erf(x)) with an approx analytic continuation for x < -1 (from Cranmer et al. 2021)
def safe_log_erf(x):
    base_mask = x < -1
    value_giving_zero = torch.zeros_like(x, device=x.device)
    x_under = torch.where(base_mask, x, value_giving_zero)
    x_over = torch.where(~base_mask, x, value_giving_zero)
    
    f_under = lambda x: (
         0.485660082730562*x + 0.643278438654541*torch.exp(x) + 
         0.00200084619923262*x**3 - 0.643250926022749 - 0.955350621183745*x**2
    )
    f_over = lambda x: torch.log(1.0 + torch.erf(x))
    
    return f_under(x_under) + f_over(x_over)

#custom loss function that uses MSE + a log(erf) term
class Loss_Func(torch.nn.Module):
    def __init__(self, maxes):
        super(Loss_Func, self).__init__()
        self.maxes = maxes

    def forward(self, predictions, targets):
        return torch.mean(safe_log_erf(self.maxes - predictions) + (predictions - targets)**2)
        
#pytorch MLP
class reg_MLP(torch.nn.Module):
    
    #initialize pytorch MLP with specified number of input/hidden/output nodes
    def __init__(self, n_feature, n_hidden, n_output, num_hidden_layers, dropout_p):
        super(reg_MLP, self).__init__()
        self.input = torch.nn.Linear(n_feature, n_hidden).to("cuda")
        self.predict = torch.nn.Linear(n_hidden, n_output).to("cuda")

        self.hiddens = []
        for i in range(num_hidden_layers-1):
            self.hiddens.append(torch.nn.Linear(n_hidden, n_hidden).to("cuda"))

        self.dropouts = []
        for i in range(num_hidden_layers):
            self.dropouts.append(torch.nn.Dropout(dropout_p).to("cuda"))

        #means and standard deviations for re-scaling inputs
        self.mass_means = np.array([-5.47074599, -5.50485362, -5.55107994])
        self.mass_stds = np.array([0.86575343, 0.86634857, 0.80166568])
        self.orb_means = np.array([1.10103604e+00, 1.34531896e+00, 1.25862804e+00, -1.44014696e+00,
                                   -1.44382469e+00, -1.48199204e+00, -1.58037780e+00, -1.59658646e+00,
                                   -1.46025210e+00, 5.35126034e-04, 2.93399827e-04, -2.07964769e-04,
                                   1.84826520e-04, 2.00942518e-04, 1.34561831e-03, -3.81075318e-02,
                                   -4.50480364e-02, -8.37049604e-02, 4.20809298e-02, 4.89242546e-02,
                                   7.81205381e-02])
        self.orb_stds = np.array([0.17770125, 0.27459303, 0.30934483, 0.60370379, 0.5976446, 0.59195887,
                                  0.68390679, 0.70470389, 0.62941292, 0.70706072, 0.70825354, 0.7082275,
                                  0.70715261, 0.70595807, 0.70598297, 0.68020376, 0.67983686, 0.66536654,
                                  0.73082135, 0.73034166, 0.73768424])
        
        self.output_means = np.array([1.20088092, 1.31667089, -1.4599554, -1.16721504, -2.10491322, -1.3807749])
        self.output_stds = np.array([0.23123815, 0.63026354, 0.51926874, 0.49942197, 0.74455827, 0.58098256])
        self.output_maxes = (np.array([50.0, 50.0, 0.0, 0.0, np.log10(np.pi), np.log10(np.pi)]) - self.output_means)/self.output_stds
        
        self.input_means = np.concatenate((self.mass_means, np.tile(self.orb_means, 100)))
        self.input_stds = np.concatenate((self.mass_stds, np.tile(self.orb_stds, 100)))

    #function to compute output pytorch tensors from input
    def forward(self, x):
        x = torch.relu(self.input(x))
        x = self.dropouts[0](x)

        for i in range(len(self.hiddens)):
            x = torch.relu(self.hiddens[i](x))
            x = self.dropouts[i+1](x)

        x = self.predict(x)
        return x

    #function to get means and stds from time series inputs
    def get_means_stds(self, inputs, min_nt=5):
        masses = inputs[:,:3]
        orb_elements = inputs[:,3:].reshape((len(inputs), 100, 21))

        if self.training:
            #add statistical noise
            nt = np.random.randint(low=min_nt, high=101) #select nt randomly from 5-100
            rand_inds = np.random.choice(100, size=nt, replace=False) #choose timesteps without replacement
            means = torch.mean(orb_elements[:,rand_inds,:], dim=1)
            stds = torch.std(orb_elements[:,rand_inds,:], dim=1)

            rand_means = torch.normal(means, stds/(nt**0.5))
            rand_stds = torch.normal(stds, stds/((2*nt - 2)**0.5))
            pooled_inputs = torch.concatenate((masses, rand_means, rand_stds), dim=1)
        else:
            #no statistical noise
            means = torch.mean(orb_elements, dim=1)
            stds = torch.std(orb_elements, dim=1)
            pooled_inputs = torch.concatenate((masses, means, stds), dim=1)

        return pooled_inputs

    #training function
    def train_model(self, Xs, Ys, learning_rate=1e-3, weight_decay=0.0, min_nt=5, epochs=1000, batch_size=2000):
        #normalize inputs and outputs
        Xs = (Xs - self.input_means)/self.input_stds
        Ys = (Ys - self.output_means)/self.output_stds

        #split training data
        x_train, x_eval, y_train, y_eval = train_test_split(Xs, Ys, test_size=0.2, shuffle=False)
        x_train_var, y_train_var = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
        x_validate, y_validate = torch.tensor(x_eval, dtype=torch.float32), torch.tensor(y_eval, dtype=torch.float32)

        #create Data Loader
        dataset = TensorDataset(x_train_var, y_train_var)
        train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        #specify loss function and optimizer
        loss_fn = Loss_Func(self.output_maxes)
        #loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        #main training loop
        lossvals, test_lossvals = np.zeros((epochs,)), np.zeros((epochs,))
        num_steps = 0
        for i in range(epochs):
            cur_losses = []
            cur_test_losses = []
            for inputs, labels in train_loader:
                #clear gradient buffers
                optimizer.zero_grad()

                #get model predictions on training batch
                self.train()
                inputs  = inputs.to("cuda")
                pooled_inputs = self.get_means_stds(inputs, min_nt=min_nt)
                output = self(pooled_inputs)

                #get model predictions on full test set
                self.eval()
                x_validate = x_validate.to("cuda")
                with torch.no_grad():
                    pooled_x_validate = self.get_means_stds(x_validate, min_nt=min_nt)
                    test_output = self(pooled_x_validate)

                #get losses
                output, labels, y_validate  = output.to("cuda"), labels.to("cuda"), y_validate.to("cuda")
                loss = loss_fn(output, labels)
                test_loss = loss_fn(test_output, y_validate)

                #get gradients with respect to parameters
                loss.backward()

                #save losses
                cur_losses.append(loss.item())
                cur_test_losses.append(test_loss.item())

                #update parameters
                optimizer.step()
                num_steps += 1

            lossvals[i] = np.mean(cur_losses)
            test_lossvals[i] = np.mean(cur_test_losses)

        return lossvals, test_lossvals, num_steps

    #function to make predictions with trained model (takes and return numpy array)
    def make_pred(self, Xs):
        self.eval()
        Xs = (Xs - self.input_means)/self.input_stds
        pooled_Xs = self.get_means_stds(torch.tensor(Xs, dtype=torch.float32))
        Ys = self(pooled_Xs).detach().numpy()
        Ys = Ys*self.output_stds + self.output_means
        
        return Ys
    
#collision outcome model class
class CollisionOrbitalOutcomeRegressor():
    #load regression model
    def __init__(self, class_model_file='collision_orbital_outcome_regressor.torch'):
        pwd = os.path.dirname(__file__)
        self.reg_model = torch.load(pwd + '/models/' + class_model_file, map_location=torch.device('cpu'))
  
    #function to run short integration
    def generate_input(self, sim, trio_inds=[1, 2, 3]):
        #get three-planet sim
        trio_sim = get_sim_copy(sim, len(sim.particles)-1, trio_inds)
        ps = trio_sim.particles
        
        #align z-axis with direction of angular momentum
        _, _ = align_simulation(trio_sim)

        #assign planet radii
        for i in range(1, len(ps)):
            ps[i].r = get_rad(ps[i].m)

        #set integration settings
        trio_sim.integrator = 'mercurius'
        trio_sim.collision = 'direct'
        trio_sim.collision_resolve = perfect_merge
        Ps = np.array([p.P for p in ps[1:len(ps)]])
        es = np.array([p.e for p in ps[1:len(ps)]])
        minTperi = np.min(Ps*(1 - es)**1.5/np.sqrt(1 + es))
        trio_sim.dt = 0.05*minTperi

        times = np.linspace(0.0, 1e4, 100)
        states = [np.log10(ps[1].m), np.log10(ps[2].m), np.log10(ps[3].m)]
        for i in range(len(times)):
            trio_sim.integrate(times[i], exact_finish_time=0)

            #check for merger
            if len(ps) == 4:
                if ps[1].inc == 0.0 or ps[2].inc == 0.0 or ps[3].inc == 0.0: #use very small inclinations to avoid -inf
                    states.extend([ps[1].a, ps[2].a, ps[3].a,
                                   np.log10(ps[1].e), np.log10(ps[2].e), np.log10(ps[3].e),
                                   -3.0, -3.0, -3.0,
                                   np.sin(ps[1].pomega), np.sin(ps[2].pomega), np.sin(ps[3].pomega),
                                   np.cos(ps[1].pomega), np.cos(ps[2].pomega), np.cos(ps[3].pomega),
                                   np.sin(ps[1].Omega), np.sin(ps[2].Omega), np.sin(ps[3].Omega),
                                   np.cos(ps[1].Omega), np.cos(ps[2].Omega), np.cos(ps[3].Omega)])
                else:
                    states.extend([ps[1].a, ps[2].a, ps[3].a, #save state
                                   np.log10(ps[1].e), np.log10(ps[2].e), np.log10(ps[3].e),
                                   np.log10(ps[1].inc), np.log10(ps[2].inc), np.log10(ps[3].inc),
                                   np.sin(ps[1].pomega), np.sin(ps[2].pomega), np.sin(ps[3].pomega),
                                   np.cos(ps[1].pomega), np.cos(ps[2].pomega), np.cos(ps[3].pomega),
                                   np.sin(ps[1].Omega), np.sin(ps[2].Omega), np.sin(ps[3].Omega),
                                   np.cos(ps[1].Omega), np.cos(ps[2].Omega), np.cos(ps[3].Omega)])
            else:
                if ps[1].a < ps[2].a:
                    return np.array([ps[1].a, ps[2].a, np.log10(ps[1].e),
                                     np.log10(ps[2].e), np.log10(ps[1].inc), np.log10(ps[2].inc)])
                else:
                    return np.array([ps[2].a, ps[1].a, np.log10(ps[2].e),
                                     np.log10(ps[1].e), np.log10(ps[2].inc), np.log10(ps[1].inc)])

        return np.array(states)
    
    #function to predict collision outcomes given one or more rebound sims
    def predict_collision_outcome(self, sims, trio_inds=None, collision_inds=None):
        #check if input is a single sim or a list of sims
        single_sim = False
        if type(sims) != list:
            sims = [sims]
            collision_inds = [collision_inds]
            single_sim = True

        #if trio_inds was not provided, assume first three planets
        if trio_inds is None:
            trio_inds = []
            for i in range(len(sims)):
                trio_inds.append([1, 2, 3])
                
        #if collision_inds was not provided, assume collisions occur between planets 1 and 2
        if collision_inds is None:
            collision_inds = []
            for i in range(len(sims)):
                collision_inds.append([1, 2])
                
        #record a1s
        a1s = np.zeros(len(sims))
        for i in range(len(sims)):
            a1s[i] = sims[i].particles[1].a
        
        mlp_inputs = []
        outcomes = []
        done_inds = []
        for i in range(len(sims)):
            out = self.generate_input(sims[i], trio_inds[i])

            if len(out) > 6:
                #re-order input states based on which two planets collide
                masses = out[:3]
                orb_elements = out[3:]
                if collision_inds[i] == [1, 2] or collision_inds[i] == [2, 1]:
                    ordered_masses = masses
                    ordered_orb_elements = orb_elements
                elif collision_inds[i] == [2, 3] or collision_inds[i] == [3, 2]:
                    ordered_masses = np.array([masses[1], masses[2], masses[0]])
                    ordered_orb_elements = np.column_stack((orb_elements[1::3], orb_elements[2::3], orb_elements[0::3])).flatten()
                elif collision_inds[i] == [1, 3] or collision_inds[i] == [3, 1]:
                    ordered_masses = np.array([masses[0], masses[2], masses[1]])
                    ordered_orb_elements = np.column_stack((orb_elements[0::3], orb_elements[2::3], orb_elements[1::3])).flatten()
                else:
                    warnings.warn('Invalid collision_inds')
                
                mlp_inputs.append(np.concatenate((ordered_masses, ordered_orb_elements)))
            else:
                #planets collided in less than 10^4 orbits
                outcomes.append(out)
                done_inds.append(i)

        if len(mlp_inputs) > 0:
            mlp_inputs = np.array(mlp_inputs)
            mlp_outcomes = self.reg_model.make_pred(mlp_inputs)

        final_outcomes = []
        j = 0 #ind for new sims array
        k = 0 #ind for mlp prediction arrays
        for i in range(len(sims)):
            if i in done_inds:
                final_outcomes.append(outcomes[j])
                j += 1
            else:
                final_outcomes.append(mlp_outcomes[k])
                k += 1
        
        #convert back to units of a1 and from log(e) -> e, log(inc) -> inc
        final_outcomes = np.array(final_outcomes)
        final_outcomes[:,0] = a1s*final_outcomes[:,0]
        final_outcomes[:,1] = a1s*final_outcomes[:,1]
        final_outcomes[:,2] = 10**final_outcomes[:,2]
        final_outcomes[:,3] = 10**final_outcomes[:,3]
        final_outcomes[:,4] = 10**final_outcomes[:,4]
        final_outcomes[:,5] = 10**final_outcomes[:,5]
        
        #re-order outputs based on semi-major axes
        for i in range(len(final_outcomes)):
            if final_outcomes[i][1] < final_outcomes[i][0]:
                final_outcomes[i] = np.array([final_outcomes[i][1], final_outcomes[i][0], final_outcomes[i][3],\
                                              final_outcomes[i][2], final_outcomes[i][5], final_outcomes[i][4]])
        
        if single_sim:
            final_outcomes = final_outcomes[0]

        return final_outcomes
