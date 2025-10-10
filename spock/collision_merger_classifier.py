import numpy as np
import os
import torch
import rebound as rb
from .tseries_feature_functions import get_collision_tseries
from .simsetup import scale_sim, replace_trio

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
    def __init__(self, model_file='collision_merger_classifier.torch', seed=None):
        self.class_model = class_MLP(45, 30, 3, 1)
        pwd = os.path.dirname(__file__)
        self.class_model.load_state_dict(torch.load(pwd + '/models/' + model_file))
        
        self.seed = seed

    def predict_collision_probs(self, sims, trio_inds=None, return_ML_inputs=False):
        """
        Predict probabilities of a collision occurring between different pairs of planets in system(s) of three (or more) planets.

        Parameters:

        sims (rebound.Simulation or list): Initial state of the multiplanet system(s).
        trio_inds (list): Indices of the three planets that make up the three-planet subset of the full system (e.g., [1, 2, 3] for the innermost three planets). Probabilities for a collision happening between planets in the trio will be predicted.
        return_ML_inputs (bool): Whether to also return the inputs for the ML model. Useful if re-using inputs with regression model, as done in the giant impact emulator (only encouraged for the painstaking user).

        Returns:

        array: Collision pair probabilities for the input system(s). If trio_inds = [i1, i2, i3], the probabilities of a collision occurring between planets i1-i2, i2-i3, and i1-i3 will be returned, in that order.
        """
        single_sim = False
        if isinstance(sims, rb.Simulation): # passed a single sim
            sims = [sims]
            if not trio_inds is None:
                trio_inds = [trio_inds]
            single_sim = True

        # if trio_inds was not provided, assume first three planets (doesn't need to be passed for three-planet systems)
        if trio_inds is None:
            trio_inds = []
            for i in range(len(sims)):
                trio_inds.append([1, 2, 3])
        sims = [scale_sim(sim, np.arange(1, sim.N)) for sim in sims] # re-scale input sims and convert units
        probs = []
        done_sims = []
        trio_sims = []
        mlp_inputs = []
        done_inds = []
        for i, sim in enumerate(sims):
            out, trio_sim, col_prob = get_collision_tseries(sim, trio_inds[i], seed=self.seed)
            if len(trio_sim.particles) == 4:
                # no merger/ejection
                mlp_inputs.append(out)

                if return_ML_inputs: # record trio_sims, if desired
                    trio_sims.append(trio_sim)
            else:
                # merger/ejection occurred
                probs.append(col_prob)
                done_inds.append(i)

                if return_ML_inputs: # record done_sims, if desired
                    done_sims.append(replace_trio(sim, trio_inds[i], trio_sim))

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

        if return_ML_inputs:
            return np.array(final_probs), [sims, trio_sims, mlp_inputs, done_sims, done_inds]

        return np.array(final_probs)

    def predict_collision_pair(self, sims, trio_inds=None, return_ML_inputs=False):
        """
        Predict which pair of planets in system(s) of three (or more) planets will collide by predicting, and then sampling, collision pair probabilities.

        Parameters:

        sims (rebound.Simulation or list): Initial state of the multiplanet system(s).
        trio_inds (list): Indices of the three planets that make up the three-planet subset of the full system (e.g., [1, 2, 3] for the innermost three planets). Collisions will be considered between each planet pair in the trio subset system.
        return_ML_inputs (bool): Whether to also return the inputs for the ML model. Useful if re-using inputs with regression model, as done in the giant impact emulator (only encouraged for the painstaking user).

        Returns:

        array: Indices for the planets in the trio predicted to be involved in the collision (e.g., [1, 2] for a collision between planets 1 and 2).
        """
        single_sim = False
        if isinstance(sims, rb.Simulation): # passed a single sim
            sims = [sims]
            if not trio_inds is None:
                trio_inds = [trio_inds]
            single_sim = True
        
        if return_ML_inputs:
            pred_probs, ML_input_data = self.predict_collision_probs(sims, trio_inds, return_ML_inputs=True)
        else:
            pred_probs = self.predict_collision_probs(sims, trio_inds, return_ML_inputs=False)

        rand_nums = np.random.rand(len(pred_probs))
        collision_inds = np.zeros((len(pred_probs), 2))
        for i, rand_num in enumerate(rand_nums):
            if rand_num < pred_probs[i][0]:
                collision_inds[i] = [1, 2]
            elif rand_num < pred_probs[i][0] + pred_probs[i][1]:
                collision_inds[i] = [2, 3]
            else:
                collision_inds[i] = [1, 3]
        if single_sim:
            collision_inds = collision_inds[0]

        if return_ML_inputs:
            return collision_inds, ML_input_data

        return collision_inds

    def cite(self):
        """
        Print citations to papers relevant to this model.
        """

        txt = r"""This paper made use of stability predictions from the Stability of Planetary Orbital Configurations Klassifier (SPOCK) package \\citep{spock}. Predictions of which planet pairs collided with one another were made using the CollisionMergerClassifier, a multilayer perceptron model (MLP) that estimates a collision probability between each pair of planets in an adjacent planetary trio from a short $10^4$-orbit N-body integration \citep{giantimpact}."""
        bib = """
@ARTICLE{spock,
   author = {{Tamayo}, Daniel and {Cranmer}, Miles and {Hadden}, Samuel and {Rein}, Hanno and {Battaglia}, Peter and {Obertas}, Alysa and {Armitage}, Philip J. and {Ho}, Shirley and {Spergel}, David N. and {Gilbertson}, Christian and {Hussain}, Naireen and {Silburt}, Ari and {Jontof-Hutter}, Daniel and {Menou}, Kristen},
    title = "{Predicting the long-term stability of compact multiplanet systems}",
  journal = {Proceedings of the National Academy of Science},
 keywords = {machine learning, dynamical systems, UAT:498, orbital dynamics, UAT:222, Astrophysics - Earth and Planetary Astrophysics},
     year = 2020,
    month = aug,
   volume = {117},
   number = {31},
    pages = {18194-18205},
      doi = {10.1073/pnas.2001258117},
archivePrefix = {arXiv},
   eprint = {2007.06521},
primaryClass = {astro-ph.EP},
   adsurl = {https://ui.adsabs.harvard.edu/abs/2020PNAS..11718194T},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{giantimpact,
}
"""
        print(txt + "\n\n\n" + bib)
