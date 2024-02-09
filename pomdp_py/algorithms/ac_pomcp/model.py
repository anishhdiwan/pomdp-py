import numpy as np
import numpy.ma as ma
import pomdp_py
from pomdp_problems.rocksample import rocksample_problem as rs
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad


# Score Estimation version of the belief network

### Belief Network ###
class LatentSpaceTf(nn.Module):
    """
    Encode/decode features into a latent space or from a latent space

    Args:
        in_dim (int): dimensionality of the input
        out_dim (int): dimensionality of the output 
        hidden_layers (list): list of the number of neurons in the hidden layers (in order of the layers)
    """

    def __init__(self, in_dim, hidden_layers, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = hidden_layers

        self.layers.insert(0, self.in_dim)
        self.layers.append(self.out_dim)
        # print(self.layers)

        modules = []
        for i in range(len(self.layers) - 1):        
            modules.append(nn.Linear(self.layers[i], self.layers[i+1]))
            if i != (len(self.layers) - 2):
                modules.append(nn.ReLU())

        self.encoder = nn.Sequential(*modules)



    def forward(self, x):
        torch.flatten(x)
        return self.encoder(x)


class FiLMEmbeddings(nn.Module):
    """
    Compute the FiLM Embedding from a conditioning input. https://arxiv.org/pdf/1709.07871.pdf

    Compute one set of scale, bias for each feature of the input
    
    Args:
        in_dim (int): dimensionality of the input
        cond_dim (int): dimensionality of the conditioning vector (cond)
        cond (tensor): conditioning tensor
        x (tensor): tensor to which conditioning is applied
    """
    def __init__(self, in_dim, cond_dim):
        super().__init__()
        self.cond_dim = cond_dim
        self.in_dim = in_dim
        self.film_encoder = nn.Sequential(
          nn.Mish(),
          nn.Linear(self.cond_dim, 2*in_dim),
          nn.Unflatten(-1, (-1, 2))
        )

    def forward(self, x, cond):
        film_encoding = self.film_encoder(cond)

        scale = film_encoding[:,0]
        bias = film_encoding[:,1]

        return scale*x + bias


class ScaleMask(nn.Module):
    """
    A very simple output un-normalisation

    Convert outputs in [0, 1] to some range [min, max]
    
    Args:
        min (float): The min value of the output range
        max (float): The max value of the output range
    """
    def __init__(self, minval, maxval):
        super().__init__()
        self.min = minval
        self.max = maxval

    def forward(self, x):
        x = self.min + (self.max - self.min)*x
        return x



class EnergyPredAutoencoder(nn.Module):
    """
    This network transforms belief subsets into an energy prediction

    Encode an input, apply FiLM conditioning, and then decode the latent space features to return an energy value

    Args:
        in_dim (int): dimensionality of the input
        latent_space_dim (int): dimensionality of the latent space
        cond_dim (int): dimensionality of the conditioning vector (cond) 
        encoder_hidden_layers (list): list of the number of neurons in the hidden layers of the encoder (in order of the layers)
        decoder_hidden_layers (list): list of the number of neurons in the hidden layers of the decoder (in order of the layers)
        batch_size (int): NOT the traditional definition of batch size. Here a batch is a belief subset. Each sample in a batch is a belief state
        unnorm_size (list): env_name, list of the unnormalisation size state features (example: size of the environment)    
    """

    def __init__(self, in_dim, latent_space_dim, cond_dim, 
        encoder_hidden_layers, decoder_hidden_layers, batch_size, unnorm_size):
        super().__init__()
        self.encoder = LatentSpaceTf(int(batch_size*in_dim), encoder_hidden_layers, latent_space_dim)
        self.film_encoder = FiLMEmbeddings(latent_space_dim, cond_dim)
        self.decoder = LatentSpaceTf(latent_space_dim, decoder_hidden_layers, 1)
        self.unnorm_size = unnorm_size

        # TODO: Change this ugly mess to something within the data processing script
        if unnorm_size[0] == "rocksample":
            n = unnorm_size[1][0]
            self.scale_mask = ScaleMask(0, n-1)
        elif unnorm_size[0] == "tag":
            max_x = unnorm_size[1][0]
            max_y = unnorm_size[1][1]
            self.scale_maskx = ScaleMask(0, max_x-1)
            self.scale_masky = ScaleMask(0, max_y-1)



    def forward(self, x, cond):
        desired_shape = x.shape
        x = x.flatten()
        x.requires_grad = True
        cond.requires_grad = True
        encoded_input = self.encoder(x)
        conditioned_input = self.film_encoder(encoded_input, cond)
        energy = self.decoder(conditioned_input)


        # score = -grad(energy(x))
        score = grad(outputs=energy, inputs=x, grad_outputs=torch.ones_like(energy), retain_graph=True, create_graph=True)[0]

        # denoised x = x - score
        new_belief = x - score

        # TODO: Change this ugly mess to something within the data processing script
        if self.unnorm_size[0] == "rocksample": 
            # Transform new_belief back to the original data modality
            new_belief = new_belief.reshape(desired_shape)
            new_belief = F.sigmoid(new_belief)
            # new_belief[:, :2] = self.scale_mask(new_belief[:, :2])

        elif self.unnorm_size[0] == "tag":
            # Transform new_belief back to the original data modality
            new_belief = new_belief.reshape(desired_shape)
            new_belief = F.sigmoid(new_belief)
            new_belief[:,0] = self.scale_maskx(new_belief[:,0])
            new_belief[:,1] = self.scale_masky(new_belief[:,1])

        
        return new_belief, energy



class BeliefProbPredicter(nn.Module):
    """
    This network takes in a belief state and predicts a vector of individual belief probabilities

    Args:
        in_dim (int): dimensionality of the input        
        hidden_layers (list): list of the number of neurons in the hidden layers of the output probability prediction head (in order of the layers)
        batch_size (int): NOT the traditional definition of batch size. Here a batch is a belief subset. Each sample in a batch is a belief state
        n (int): size of the environment    
    """
    def __init__(self, in_dim, hidden_layers, batch_size):
        super().__init__()
        # Take in the whole batch of concatenated outputs and compute their probabilities
        self.output_prob_predicter = nn.Sequential(
            LatentSpaceTf(int(batch_size*in_dim), hidden_layers, batch_size),
            nn.Softmax(dim=0)
            )

    def forward(self, x):
        output_probabilities = self.output_prob_predicter(x.flatten())
        return output_probabilities


### Q-Network ###
class dueling_net(nn.Module):
    """
    Returns the Q(a|h)

    Modified dueling network architecture from "Dueling Network Architectures for Deep Reinforcement Learning"
    Implementation modified from https://nn.labml.ai/rl/dqn/model.html


    Args:
        n_actions (int): number of actions in the env
        in_dim (int): dimensionality of the input
        latent_space_dim (int): dimensionality of the latent space of the encoding 
        encoder_hidden_layers (list): list of the number of neurons in the hidden layers of the encoder (in order of the layers)
        state_decoder_hidden_layers (list): list of the number of neurons in the hidden layers of the state decoder (in order of the layers)
        advantage_decoder_hidden_layers (list): list of the number of neurons in the hidden layers of the advantage decoder (in order of the layers)
    """
    def __init__(self, n_actions, in_dim, encoder_hidden_layers, latent_space_dim, 
        state_decoder_hidden_layers, advantage_decoder_hidden_layers):
        super(dueling_net, self).__init__()

        self.encoder = LatentSpaceTf(in_dim, encoder_hidden_layers, latent_space_dim)

        # This head gives the state value $V$
        self.state_value_decoder = LatentSpaceTf(latent_space_dim, state_decoder_hidden_layers, 1)

        # This head gives the advantage value $A$
        self.advantage_value_decoder = LatentSpaceTf(latent_space_dim, advantage_decoder_hidden_layers, n_actions)


    def forward(self, x):
        # Convolution
        encoded_input = self.encoder(x)

        # $A$
        advantage_value = self.advantage_value_decoder(encoded_input)
        # $V$
        state_value = self.state_value_decoder(encoded_input)

        action_score_centered = advantage_value - advantage_value.mean(dim=-1, keepdim=True)
        q = state_value + action_score_centered

        return q



class Network_Utils():
    """
    Utility functions for the qvalue and belief networks

    Args:
        energy_net (nn.Module): Belief network
        q_net (nn.Module): Q value network
        env_data_processing (class): Data processing utils for the environment
        bel_prob (ndarray): Probabilities of each belief state
        qnet_lr (float): q network learning rate
        belnet_lr (float): belief network learning rate
        init_bel_prob (ndarray): initial belief probabilities

    """
    def __init__(self, energy_net, bel_prob_net, q_net, env_data_processing, init_bel_prob, qnet_lr, belprobnet_lr, energynet_lr, discount_factor):
        self.energy_net = energy_net.to(torch.float)
        self.bel_prob_net = bel_prob_net.to(torch.float)
        self.q_net = q_net.to(torch.float)
        self.env_data_processing = env_data_processing
        self.bel_prob = init_bel_prob
        self.qnet_lr = qnet_lr
        self.belprobnet_lr = belprobnet_lr
        self.energynet_lr = energynet_lr
        self.hist_tensor = None
        self.true_next_state = None # The true updated state of the env. Contains observable features that are used in the belief update
        self.discount_factor = discount_factor

        self.qnet_optim = optim.Adam(self.q_net.parameters(), lr=self.qnet_lr, weight_decay=1e-5) # Weight decay is L2 regularization
        self.energynet_optim = optim.Adam(self.energy_net.parameters(), lr=self.energynet_lr, weight_decay=1e-5)
        self.belprobnet_optim = optim.Adam(self.bel_prob_net.parameters(), lr=self.belprobnet_lr, weight_decay=1e-5)
    
    def argmax(self, hist_conditioned_qvalues):
        # Creating a clone to convert nan to num. Nans are needed during the update later on but affect the argmax operation
        temp_qvalues = torch.clone(hist_conditioned_qvalues).detach()
        idx = torch.argmax(torch.nan_to_num(temp_qvalues), dim=0)
        actions_dict = dict((v, k) for k, v in self.env_data_processing.actions.items())
        action_name = actions_dict[idx.item()]
        action_value = hist_conditioned_qvalues[idx.item()]
        return action_name, action_value

    def getHistoryConditionedQValues(self, bel_state_conditioned_qvalues, probabilities):
        # Get Q(.|h) = SUM (p(s) * Q(.|s))
        # bel_state_conditioned_qvalues is a dictionary of {action: value} pairs for each particle in the belief
        bel_state_conditioned_qvalues = self.env_data_processing.qval_array_from_dict(bel_state_conditioned_qvalues)
        probabilities = probabilities.view(bel_state_conditioned_qvalues.shape[0], -1)
        hist_conditioned_qvalues = (probabilities * bel_state_conditioned_qvalues).sum(dim=0)
        self.hist_conditioned_qvalues = hist_conditioned_qvalues

        return hist_conditioned_qvalues


    def getHNextValue(self, bel_state_conditioned_hnextvalues, action_name, real_observation, probabilities):
        # Given a list of dicts with action: dict(obs: value), get the hnext values conditioned on the current history. Then return their weighted avg
        hnext_values = np.zeros(len(bel_state_conditioned_hnextvalues))
        for idx, hnext_dict in enumerate(bel_state_conditioned_hnextvalues):
            # Add a value of 0.0 if either the action was not taken in this belief state's tree or if the observation was not seen after that action
            if action_name in hnext_dict:
                hnext_values[idx] = hnext_dict[action_name].get(str(real_observation), 0.0)
            else:
                hnext_values[idx] = 0.0

        # Gradient backprop is not necessary for next history q values (because of the semi-gradient update). Converting the tensor to np array if needed
        if torch.is_tensor(probabilities):
            bel_prob = probabilities.detach().clone().numpy()
        else:
            bel_prob = probabilities
        hnext_value = (bel_prob * hnext_values).sum()


        return hnext_value


    def getNewBelief(self, agent, first_step=False):
        # History is set only after getNewBelief() is executed. This is not true on step 1 as the uniform belief is used. Hence it is added here for the first update
        if first_step:
            self.hist_tensor = self.env_data_processing.cond_from_history(agent.history)
            belief_tensor = self.env_data_processing.batch_from_particles(belief=agent.belief)
            _, self.energy = self.energy_net(belief_tensor.to(torch.float), self.hist_tensor)
        
        else:
            # Compute the new belief state and probabilities given the old ones
            belief_tensor = self.env_data_processing.batch_from_particles(belief=agent.belief)
            self.hist_tensor = self.env_data_processing.cond_from_history(agent.history)
            
            new_belief, self.energy = self.energy_net(belief_tensor.to(torch.float), self.hist_tensor)
            # Detach the new belief so that gradients do not flow back to the energy network
            new_belief_detached = new_belief.clone().detach()
            new_bel_prob = self.bel_prob_net(new_belief_detached)

            self.bel_prob = new_bel_prob

            new_belief = self.env_data_processing.particles_from_output(new_belief, self.true_next_state)
            return new_belief, new_bel_prob


    def updateNetworks(self, agent, reward, best_action_value, hnext_value):
        ## Note: Switching from using predicted Q values in the semi-gradient update to using hnext values from the search tree.
        ## Still keeping the q network for comparison! It is however NOT USED Anywhere


        pred_q_values = self.q_net(self.hist_tensor)
        hist_conditioned_qvalues = self.hist_conditioned_qvalues
        best_action_value = torch.tensor([best_action_value], requires_grad=True)
        reward_tensor = torch.tensor([reward], requires_grad=False)

        assert pred_q_values.shape == hist_conditioned_qvalues.shape, "The shapes of Q(.|h) and Qnet(h) must match"

        mask = torch.isnan(hist_conditioned_qvalues)
        hist_conditioned_qvalues[mask] = 0.
        pred_q_values[mask] = 0.


        ### QNET LOSS ### 
        qnet_loss = F.mse_loss(pred_q_values.to(torch.float), hist_conditioned_qvalues)

        ### BELIEF PROBABILITY NET LOSS ###
        # Assuming that the agent's history agent.history has been updated via agent.update_history() and a reward has been seen via env.state_transition()
        # next_hist_tensor = self.env_data_processing.cond_from_history(agent.history)
        # best_next_action = torch.max(self.q_net(next_hist_tensor))

        ## Replacing q net prediction with search tree next hist values 
        best_next_action = torch.tensor([hnext_value], requires_grad=False).to(torch.float)

        # delta = R + gamma*max(Q(hnext)) - Q(areal) 
        # Here R + gamma*max(Q(hnext)) is the return in bootstrap form. We are trying to change Q(areal) so that it is close to the return
        # Q(areal) = argmax( Q(.|h) ) = argmax( bel_prob * some_vector_from_sim ) = argmax( bel_net(old_bet, hist) * some_vector )
        # This means backward() will compute gradients for the bel_net 
        # requires_grad=False for bootstrapped_return since we do not want to update the Q_network in this step. We only want to backprop gradients to the belief network
        bootstrapped_return = reward + self.discount_factor*best_next_action
        bootstrapped_return = torch.tensor([bootstrapped_return], requires_grad=False) 

        belprobnet_loss = F.mse_loss(bootstrapped_return, best_action_value)

        ### ENERGY NET LOSS ###
        energynet_loss = F.mse_loss(self.energy, reward_tensor)

        ### UPDATE ###
        # Not used anywhere! Just here for comparison
        self.qnet_optim.zero_grad()
        qnet_loss.backward()
        self.qnet_optim.step()

        self.belprobnet_optim.zero_grad()
        belprobnet_loss.backward()
        self.belprobnet_optim.step()

        self.energynet_optim.zero_grad()
        energynet_loss.backward()
        self.energynet_optim.step()


        return qnet_loss, belprobnet_loss, energynet_loss



