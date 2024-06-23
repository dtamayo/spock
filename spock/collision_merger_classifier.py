import numpy as np
import os
import torch
from .tseries_feature_functions import get_collision_tseries

# pytorch MLP class
class class_MLP(torch.nn.Module):
    
    # initialize MLP with specified number of input/hidden/output nodes
    def __init__(self, n_feature, n_hidden, n_output, num_hidden_layers):
        super(class_MLP, self).__init__()
        self.input = torch.nn.Linear(n_feature, n_hidden).to("cpu")
        self.predict = torch.nn.Linear(n_hidden, n_output).to("cpu")

        self.hiddens = []
        for i in range(num_hidden_layers-1):
            self.hiddens.append(torch.nn.Linear(n_hidden, n_hidden).to("cpu"))

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
        means = torch.mean(orb_elements, dim=1)
        stds = torch.std(orb_elements, dim=1)
        pooled_inputs = torch.concatenate((masses, means, stds), dim=1)

        return pooled_inputs

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
    def __init__(self, model_file='collision_merger_classifier.torch'):
        self.class_model = class_MLP(45, 30, 3, 1)
        pwd = os.path.dirname(__file__)
        self.class_model.load_state_dict(torch.load(pwd + '/models/' + model_file))
        
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
        trio_sims = []
        for i, sim in enumerate(sims):
            out, trio_sim = get_collision_tseries(sim, trio_inds[i]) 
            
            if len(trio_sim.particles) == 4:
                # no merger/ejection
                mlp_inputs.append(out)
            elif len(trio_sim.particles) == 3:
                # check for ejection or merger
                ps = trio_sim.particles
                if np.log10(ps[1].m) in [out[0], out[1], out[2]] and np.log10(ps[2].m) in [out[0], out[1], out[2]]: #ejection
                    probs.append(np.array([0.0, 0.0, 0.0]))
                elif out[0] == np.log10(ps[1].m) or out[0] == np.log10(ps[2].m): # planets 2 and 3 merged
                    probs.append(np.array([0.0, 1.0, 0.0]))
                elif out[1] == np.log10(ps[1].m) or out[1] == np.log10(ps[2].m): # planets 1 and 3 merged
                    probs.append(np.array([0.0, 0.0, 1.0]))
                elif out[2] == np.log10(ps[1].m) or out[2] == np.log10(ps[2].m): # planets 1 and 2 merged
                    probs.append(np.array([1.0, 0.0, 0.0]))
                done_inds.append(i)
            else:
                probs.append(np.array([0.0, 0.0, 0.0]))
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

    def sample_collision_indices(self, sims, trio_inds=None):
        pred_probs = self.predict_collision_probs(sims, trio_inds)
        rand_nums = np.random.rand(len(pred_probs))
        collision_inds = np.zeros((len(pred_probs), 2))
        for i, rand_num in enumerate(rand_nums):
            if rand_num < pred_probs[i][0]:
                collision_inds[i] = [1, 2]
            elif rand_num < pred_probs[i][0] + pred_probs[i][1]:
                collision_inds[i] = [2, 3]
            else:
                collision_inds[i] = [1, 3]
        
        return collision_inds
