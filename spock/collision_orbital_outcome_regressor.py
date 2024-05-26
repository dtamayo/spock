import numpy as np
import random
import os
import torch
import warnings
import rebound as rb
from .simsetup import copy_sim, align_simulation, get_rad, perfect_merge, replace_trio, revert_sim_units
from .tseries_feature_functions import get_collision_tseries

# pytorch MLP
class reg_MLP(torch.nn.Module):
    
    # initialize pytorch MLP with specified number of input/hidden/output nodes
    def __init__(self, n_feature, n_hidden, n_output, num_hidden_layers):
        super(reg_MLP, self).__init__()
        self.input = torch.nn.Linear(n_feature, n_hidden).to("cpu")
        self.predict = torch.nn.Linear(n_hidden, n_output).to("cpu")

        self.hiddens = []
        for i in range(num_hidden_layers-1):
            self.hiddens.append(torch.nn.Linear(n_hidden, n_hidden).to("cpu"))

        # means and standard deviations for re-scaling inputs
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

    # function to compute output pytorch tensors from input
    def forward(self, x):
        x = torch.relu(self.input(x))

        for hidden_layer in self.hiddens:
             x = torch.relu(hidden_layer(x))

        x = self.predict(x)
        return x

    # function to get means and stds from time series inputs
    def get_means_stds(self, inputs, min_nt=5):
        masses = inputs[:,:3]
        orb_elements = inputs[:,3:].reshape((len(inputs), 100, 21))
        means = torch.mean(orb_elements, dim=1)
        stds = torch.std(orb_elements, dim=1)
        pooled_inputs = torch.concatenate((masses, means, stds), dim=1)

        return pooled_inputs

    # function to make predictions with trained model (takes and return numpy array)
    def make_pred(self, Xs):
        self.eval()
        Xs = (Xs - self.input_means)/self.input_stds
        pooled_Xs = self.get_means_stds(torch.tensor(Xs, dtype=torch.float32))
        Ys = self(pooled_Xs).detach().numpy()
        Ys = Ys*self.output_stds + self.output_means
        
        return Ys
    
# collision outcome model class
class CollisionOrbitalOutcomeRegressor():
    # load regression model
    def __init__(self, model_file='collision_orbital_outcome_regressor.torch', seed=None):
        self.reg_model = reg_MLP(45, 60, 6, 1)
        pwd = os.path.dirname(__file__)
        self.reg_model.load_state_dict(torch.load(pwd + '/models/' + model_file))
        
        # set random seed
        if not seed is None:
            os.environ["PL_GLOBAL_SEED"] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    # function to predict collision outcomes given one or more rebound sims
    def predict_collision_outcome(self, sims, trio_inds=None, collision_inds=None):
        # check if input is a single sim or a list of sims
        single_sim = False
        if type(sims) != list:
            sims = [sims]
            collision_inds = [collision_inds]
            single_sim = True

        # if trio_inds was not provided, assume first three planets
        if trio_inds is None:
            trio_inds = []
            for i in range(len(sims)):
                trio_inds.append([1, 2, 3])
                
        # if collision_inds was not provided, assume collisions occur between planets 1 and 2
        if collision_inds is None:
            collision_inds = []
            for i in range(len(sims)):
                collision_inds.append([1, 2])
                
        # record units and G of input sims
        original_units = sims[0].units
        original_G = sims[0].G
        
        # record original M_stars, a1s, and P1s
        original_Mstars = np.zeros(len(sims))
        original_a1s = np.zeros(len(sims))
        original_P1s = np.zeros(len(sims))
        for i, sim in enumerate(sims):
            original_Mstars[i] = sim.particles[0].m
            original_a1s[i] = sim.particles[1].a
            original_P1s[i] = sim.particles[1].P
        
        # re-scale input sims and convert units
        sims = [copy_sim(sim, np.arange(1, sim.N), scaled=True) for sim in sims]
        
        mlp_inputs = []
        thetas = []
        done_sims = []
        done_inds = []
        for i, sim in enumerate(sims):
            out, trio_sim, theta1, theta2 = get_collision_tseries(sim, trio_inds[i])
            
            if len(trio_sim.particles) == 4:
                # no merger (or ejection)
                mlp_inputs.append(out)
                thetas.append([theta1, theta2])
            else: 
                # if merger/ejection occurred, save sim
                done_sims.append(replace_trio(sim, trio_inds[i], trio_sim, theta1, theta2))
                done_inds.append(i)
                
        if len(mlp_inputs) > 0:
            # re-order input array based on input collision_inds
            reg_inputs = []
            for i, col_ind in enumerate(collision_inds):
                masses = mlp_inputs[i][:3]
                orb_elements = mlp_inputs[i][3:]

                if (col_ind[0] == 1 and col_ind[1] == 2) or (col_ind[0] == 2 and col_ind[1] == 1): # merge planets 1 and 2
                    ordered_masses = masses
                    ordered_orb_elements = orb_elements
                elif (col_ind[0] == 2 and col_ind[1] == 3) or (col_ind[0] == 3 and col_ind[1] == 2): # merge planets 2 and 3
                    ordered_masses = np.array([masses[1], masses[2], masses[0]])
                    ordered_orb_elements = np.column_stack((orb_elements[1::3], orb_elements[2::3], orb_elements[0::3])).flatten()
                elif (col_ind[0] == 1 and col_ind[1] == 3) or (col_ind[0] == 3 and col_ind[1] == 1): # merge planets 1 and 3
                    ordered_masses = np.array([masses[0], masses[2], masses[1]])
                    ordered_orb_elements = np.column_stack((orb_elements[0::3], orb_elements[2::3], orb_elements[1::3])).flatten()
                else:
                    warnings.warn('Invalid collision_inds')

                reg_inputs.append(np.concatenate((ordered_masses, ordered_orb_elements)))

            # predict orbital elements with regression model
            reg_inputs = np.array(reg_inputs)
            reg_outputs = self.reg_model.make_pred(reg_inputs)

            m1s = 10**reg_inputs[:,0] + 10**reg_inputs[:,1] # new planet
            m2s = 10**reg_inputs[:,2] # surviving planet
            a1s = reg_outputs[:,0]
            a2s = reg_outputs[:,1]
            e1s = 10**reg_outputs[:,2]
            e2s = 10**reg_outputs[:,3]
            inc1s = 10**reg_outputs[:,4]
            inc2s = 10**reg_outputs[:,5]
            
        new_sims = []
        j = 0 # index for new sims array
        k = 0 # index for mlp prediction arrays
        for i in range(len(sims)):
            if i in done_inds:
                new_sims.append(done_sims[j])
                j += 1
            else:                
                # create sim that contains state of two predicted planets
                new_state_sim = rb.Simulation()
                new_state_sim.G = 4*np.pi**2 # units in which a1=1.0 and P1=1.0
                new_state_sim.add(m=1.00)
                try:
                    new_state_sim.add(m=m1s[k], a=a1s[k], e=e1s[k], inc=inc1s[k], pomega=np.random.uniform(0.0, 2*np.pi), Omega=np.random.uniform(0.0, 2*np.pi), l=np.random.uniform(0.0, 2*np.pi))
                except Exception as e:
                    warnings.warn('Removing planet with unphysical orbital elements')
                try:
                    new_state_sim.add(m=m2s[k], a=a2s[k], e=e2s[k], inc=inc2s[k], pomega=np.random.uniform(0.0, 2*np.pi), Omega=np.random.uniform(0.0, 2*np.pi), l=np.random.uniform(0.0, 2*np.pi))
                except Exception as e:
                    warnings.warn('Removing planet with unphysical orbital elements')
                new_state_sim.move_to_com()

                # replace trio with predicted duo (or single/zero if planets have unphysical orbital elements)
                new_sims.append(replace_trio(sims[i], trio_inds[i], new_state_sim, thetas[k][0], thetas[k][1]))
                k += 1
        
        # convert sims back to original units
        new_sims = revert_sim_units(new_sims, original_Mstars, original_a1s, original_G, original_units)
        
        if single_sim:
            new_sims = new_sims[0]
        
        return new_sims
