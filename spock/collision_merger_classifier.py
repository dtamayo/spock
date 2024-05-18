import numpy as np
import os
import torch
from .simsetup import get_sim_copy, align_simulation, get_rad, perfect_merge

# pytorch MLP class
class class_MLP(torch.nn.Module):
    
    # initialize MLP with specified number of input/hidden/output nodes
    def __init__(self, n_feature, n_hidden, n_output, num_hidden_layers):
        super(class_MLP, self).__init__()
        self.input = torch.nn.Linear(n_feature, n_hidden).to("cuda")
        self.predict = torch.nn.Linear(n_hidden, n_output).to("cuda")

        self.hiddens = []
        for i in range(num_hidden_layers-1):
            self.hiddens.append(torch.nn.Linear(n_hidden, n_hidden).to("cuda"))

        # means and standard deviations for re-scaling inputs
        self.mass_means = np.array([-5.47727975, -5.58391119, -5.46548861])
        self.mass_stds = np.array([0.85040165, 0.82875662, 0.85292227])
        self.orb_means = np.array([1.00610835e+00, 1.22315510e+00, 1.47571958e+00, -1.45349794e+00,
                                   -1.42549269e+00, -1.48697306e+00, -1.54294123e+00, -1.49154390e+00,
                                   -1.60273122e+00, 3.28216683e-04, 6.35070370e-05, 2.28837372e-04,
                                   7.22626143e-04, 5.37250147e-04, 4.71511054e-04, -5.73411601e-02,
                                   -5.63092298e-02, -5.32101388e-02, 5.75283781e-02, 4.83608439e-02,
                                   6.32365005e-02])
        self.orb_stds = np.array([0.06681375, 0.180157, 0.29317225, 0.61093399, 0.57057764, 0.61027233,
                                  0.67640293, 0.63564565, 0.7098103, 0.70693578, 0.70823902, 0.70836691,
                                  0.7072773, 0.70597252, 0.70584421, 0.67243801, 0.68578479, 0.66805109,
                                  0.73568308, 0.72400948, 0.7395117])
        
        self.input_means = np.concatenate((self.mass_means, np.tile(self.orb_means, 100)))
        self.input_stds = np.concatenate((self.mass_stds, np.tile(self.orb_stds, 100)))

    # function to compute output pytorch tensors from input
    def forward(self, x):
        x = torch.relu(self.input(x))

        for hidden_layer in self.hiddens:
             x = torch.relu(hidden_layer(x))

        x = torch.softmax(self.predict(x), dim=1)
        return x

    # function to get means and stds from time series inputs
    def get_means_stds(self, inputs, min_nt=5):
        masses = inputs[:,:3]
        orb_elements = inputs[:,3:].reshape((len(inputs), 100, 21))

        if self.training:
            # add statistical noise
            nt = np.random.randint(low=min_nt, high=101) #select nt randomly from 5-100
            rand_inds = np.random.choice(100, size=nt, replace=False) #choose timesteps without replacement
            means = torch.mean(orb_elements[:,rand_inds,:], dim=1)
            stds = torch.std(orb_elements[:,rand_inds,:], dim=1)

            rand_means = torch.normal(means, stds/(nt**0.5))
            rand_stds = torch.normal(stds, stds/((2*nt - 2)**0.5))
            pooled_inputs = torch.concatenate((masses, rand_means, rand_stds), dim=1)
        else:
            # no statistical noise
            means = torch.mean(orb_elements, dim=1)
            stds = torch.std(orb_elements, dim=1)
            pooled_inputs = torch.concatenate((masses, means, stds), dim=1)

        return pooled_inputs

    # training function
    def train_model(self, Xs, Ys, learning_rate=1e-3, weight_decay=0.0, min_nt=5, epochs=1000, batch_size=2000):
        # normalize inputs
        Xs = (Xs - self.input_means)/self.input_stds

        # split training data
        x_train, x_eval, y_train, y_eval = train_test_split(Xs, Ys, test_size=0.2, shuffle=False)
        x_train_var, y_train_var = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
        x_validate, y_validate = torch.tensor(x_eval, dtype=torch.float32), torch.tensor(y_eval, dtype=torch.float32)

        # create Data Loader
        dataset = TensorDataset(x_train_var, y_train_var)
        train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        # specify loss function and optimizer
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # main training loop
        lossvals, test_lossvals = np.zeros((epochs,)), np.zeros((epochs,))
        num_steps = 0
        for i in range(epochs):
            cur_losses = []
            cur_test_losses = []
            for inputs, labels in train_loader:
                # clear gradient buffers
                optimizer.zero_grad()

                # get model predictions on training batch
                self.train()
                inputs  = inputs.to("cuda")
                pooled_inputs = self.get_means_stds(inputs, min_nt=min_nt)
                output = self(pooled_inputs)

                # get model predictions on full test set
                self.eval()
                x_validate = x_validate.to("cuda")
                with torch.no_grad():
                    pooled_x_validate = self.get_means_stds(x_validate, min_nt=min_nt)
                    test_output = self(pooled_x_validate)

                # get losses
                output, labels, y_validate  = output.to("cuda"), labels.to("cuda"), y_validate.to("cuda")
                loss = loss_fn(output, labels)
                test_loss = loss_fn(test_output, y_validate)

                # get gradients with respect to parameters
                loss.backward()

                # save losses
                cur_losses.append(loss.item())
                cur_test_losses.append(test_loss.item())

                # update parameters
                optimizer.step()
                num_steps += 1

            lossvals[i] = np.mean(cur_losses)
            test_lossvals[i] = np.mean(cur_test_losses)

        return lossvals, test_lossvals, num_steps

    # function to make predictions with trained model (takes and returns numpy array)
    def make_pred(self, Xs):
        self.eval()
        Xs = (Xs - self.input_means)/self.input_stds
        pooled_Xs = self.get_means_stds(torch.tensor(Xs, dtype=torch.float32))
        Ys = self(pooled_Xs).detach().numpy()

        return Ys

# collision classification model class
class CollisionMergerClassifier():
    
    # load classification model
    def __init__(self, class_model_file='collision_merger_classifier.torch'):
        pwd = os.path.dirname(__file__)
        self.class_model = torch.load(pwd + '/models/' + class_model_file, map_location=torch.device('cpu'))
  
    # function to run short integration
    def generate_input(self, sim, trio_inds=[1, 2, 3]):
        # get three-planet sim
        trio_sim = get_sim_copy(sim, trio_inds)
        ps = trio_sim.particles
        
        # align z-axis with direction of angular momentum
        _, _ = align_simulation(trio_sim)

        # assign planet radii
        for i in range(1, len(ps)):
            ps[i].r = get_rad(ps[i].m)

        # set integration settings
        trio_sim.integrator = 'mercurius'
        trio_sim.collision = 'direct'
        trio_sim.collision_resolve = perfect_merge
        Ps = np.array([p.P for p in ps[1:len(ps)]])
        es = np.array([p.e for p in ps[1:len(ps)]])
        minTperi = np.min(Ps*(1 - es)**1.5/np.sqrt(1 + es))
        trio_sim.dt = 0.05*minTperi

        times = np.linspace(0.0, 1e4, 100)
        states = [np.log10(ps[1].m), np.log10(ps[2].m), np.log10(ps[3].m)]
        
        for t in times:
            trio_sim.integrate(t, exact_finish_time=0)

            # check for merger
            if len(ps) == 4:
                if ps[1].inc == 0.0 or ps[2].inc == 0.0 or ps[3].inc == 0.0:
                    # use very small inclinations to avoid -inf
                    states.extend([ps[1].a, ps[2].a, ps[3].a,
                                   np.log10(ps[1].e), np.log10(ps[2].e), np.log10(ps[3].e),
                                   -3.0, -3.0, -3.0,
                                   np.sin(ps[1].pomega), np.sin(ps[2].pomega), np.sin(ps[3].pomega),
                                   np.cos(ps[1].pomega), np.cos(ps[2].pomega), np.cos(ps[3].pomega),
                                   np.sin(ps[1].Omega), np.sin(ps[2].Omega), np.sin(ps[3].Omega),
                                   np.cos(ps[1].Omega), np.cos(ps[2].Omega), np.cos(ps[3].Omega)])
                else:
                    states.extend([ps[1].a, ps[2].a, ps[3].a,
                                   np.log10(ps[1].e), np.log10(ps[2].e), np.log10(ps[3].e),
                                   np.log10(ps[1].inc), np.log10(ps[2].inc), np.log10(ps[3].inc),
                                   np.sin(ps[1].pomega), np.sin(ps[2].pomega), np.sin(ps[3].pomega),
                                   np.cos(ps[1].pomega), np.cos(ps[2].pomega), np.cos(ps[3].pomega),
                                   np.sin(ps[1].Omega), np.sin(ps[2].Omega), np.sin(ps[3].Omega),
                                   np.cos(ps[1].Omega), np.cos(ps[2].Omega), np.cos(ps[3].Omega)])
            else:
                if states[0] == np.log10(ps[1].m) or states[0] == np.log10(ps[2].m): #planets 2 and 3 merged
                    return np.array([0.0, 1.0, 0.0])
                elif states[1] == np.log10(ps[1].m) or states[1] == np.log10(ps[2].m): #planets 1 and 3 merged
                    return np.array([0.0, 0.0, 1.0])
                elif states[2] == np.log10(ps[1].m) or states[2] == np.log10(ps[2].m): #planets 1 and 2 merged
                    return np.array([1.0, 0.0, 0.0])

        return np.array(states)
        
    # function to predict collision probabilities given one or more rebound sims
    def predict_collision_probs(self, sims, trio_inds=None):
        # check if input is a single sim or a list of sims
        single_sim = False
        if type(sims) != list:
            sims = [sims]
            single_sim = True

        # if trio_inds was not provided, assume first three planets
        if trio_inds is None:
            trio_inds = []
            for i in range(len(sims)):
                trio_inds.append([1, 2, 3])

        mlp_inputs = []
        probs = []
        done_inds = []
        
        for i, sim in enumerate(sims):
            out = self.generate_input(sim, trio_inds[i])

            if len(out) > 3:
                mlp_inputs.append(out)
            else:
                # planets collided in less than 10^4 orbits
                probs.append(out)
                done_inds.append(i)

        if len(mlp_inputs) > 0:
            mlp_inputs = np.array(mlp_inputs)
            mlp_probs = self.class_model.make_pred(mlp_inputs)

        final_probs = []
        j = 0 # index for new sims array
        k = 0 # index for mlp prediction arrays
        for i in range(len(sims)):
            if i in done_inds:
                final_probs.append(probs[j])
                j += 1
            else:
                final_probs.append(mlp_probs[k])
                k += 1
                
        if single_sim:
            final_probs = final_probs[0]

        return np.array(final_probs)
