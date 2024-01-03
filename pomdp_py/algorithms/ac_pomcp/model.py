import numpy as np
import numpy.ma as ma
import pomdp_py
from pomdp_problems.rocksample import rocksample_problem as rs
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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


class BoolMask(nn.Module):
    """
    A very simple boolean mask.

    Compare each element with the mask value and output a new boolean tensor
    
    Args:
        maskval (float): The threshold to compare against
    """
    def __init__(self, maskval=0.5):
        super().__init__()
        self.maskval = maskval

    def forward(self, x):
        x = x < self.maskval
        return x


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



class MultiHeadAutoencoder(nn.Module):
    """
    This network transforms belief subsets into new belief subsets.

    Encode an input, apply FiLM conditioning, and then decode the latent space features via multiple heads

    Args:
        in_dim (int): dimensionality of the input
        latent_space_dim (int): dimensionality of the latent space
        cond_dim (int): dimensionality of the conditioning vector (cond)
        out_dims (list): list of dimensionalities of the output of each head 
        encoder_hidden_layers (list): list of the number of neurons in the hidden layers of the encoder (in order of the layers)
        decoder_hidden_layers (2D list): 2D list of the number of neurons in the hidden layers of the decoder (in order of the layers)
        output_prob_predicter_hidden_layers (list): list of the number of neurons in the hidden layers of the output probability prediction head (in order of the layers)
        batch_size (int): NOT the traditional definition of batch size. Here a batch is a belief subset. Each sample in a batch is a belief state
        n (int): size of the environment    
    """

    def __init__(self, in_dim, latent_space_dim, cond_dim, out_dims, 
        encoder_hidden_layers, decoder_hidden_layers, output_prob_predicter_hidden_layers, batch_size, n):
        super().__init__()
        self.encoder = LatentSpaceTf(in_dim, encoder_hidden_layers, latent_space_dim)
        self.film_encoder = FiLMEmbeddings(latent_space_dim, cond_dim)

        # Output heads
        self.output_heads = []
        for i in range(len(out_dims)):
            self.output_heads.append([LatentSpaceTf(latent_space_dim, decoder_hidden_layers[i], out_dims[i]), nn.Sigmoid()])

        self.output_heads[0].append(ScaleMask(0, n-1))
        # self.output_heads[1].append(BoolMask(0.5))
        # self.output_heads[2].append(BoolMask(0.5))


        for i in range(len(self.output_heads)):
            self.output_heads[i] = nn.Sequential(*self.output_heads[i])


        # Take in the whole batch of concatenated outputs and compute their probabilities
        self.output_prob_predicter = nn.Sequential(
            LatentSpaceTf(int(batch_size*sum(out_dims)), output_prob_predicter_hidden_layers, batch_size),
            nn.Softmax(dim=0)
            )


    def forward(self, x, cond):
        encoded_input = self.encoder(x)
        conditioned_input = self.film_encoder(encoded_input, cond)

        outputs = []
        for output_head in self.output_heads:
            outputs.append(output_head(conditioned_input))

        concatenated_output = torch.cat((outputs), 1).to(torch.float)
        output_probabilities = self.output_prob_predicter(concatenated_output.flatten())
        
        return concatenated_output, output_probabilities


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
        belief_net (nn.Module): Belief network
        q_net (nn.Module): Q value network
        env_data_processing (class): Data processing utils for the environment
        bel_prob (ndarray): Probabilities of each belief state
        qnet_lr (float): q network learning rate
        belnet_lr (float): belief network learning rate
        init_bel_prob (ndarray): initial belief probabilities

    """
    def __init__(self, belief_net, q_net, env_data_processing, init_bel_prob, qnet_lr, belnet_lr, discount_factor):
        self.belief_net = belief_net.to(torch.float)
        self.q_net = q_net.to(torch.float)
        self.env_data_processing = env_data_processing
        self.bel_prob = init_bel_prob
        self.qnet_lr = qnet_lr
        self.belnet_lr = belnet_lr
        self.hist_tensor = None
        self.discount_factor = discount_factor

        self.qnet_optim = optim.Adam(self.q_net.parameters(), lr=self.qnet_lr, weight_decay=1e-5) # Weight decay is L2 regularization
        self.belnet_optim = optim.Adam(self.belief_net.parameters(), lr=self.belnet_lr, weight_decay=1e-5) # Weight decay is L2 regularization

    
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
        # print(bel_state_conditioned_qvalues)
        # print(probabilities.view(bel_state_conditioned_qvalues.shape[0], -1))
        probabilities = probabilities.view(bel_state_conditioned_qvalues.shape[0], -1)
        # hist_conditioned_qvalues = np.nan_to_num(np.average(bel_state_conditioned_qvalues, axis=0, weights=probabilities))
        hist_conditioned_qvalues = (probabilities * bel_state_conditioned_qvalues).sum(dim=0)
        self.hist_conditioned_qvalues = hist_conditioned_qvalues

        return hist_conditioned_qvalues

    def getNewBelief(self, agent, first_step=False):
        # History is set only after getNewBelief() is executed. This is not true on step 1 as the uniform belief is used. Hence it is added here for the first update
        if first_step:
            self.hist_tensor = self.env_data_processing.cond_from_history(agent.history)
        
        else:
            # Compute the new belief state and probabilities given the old ones
            belief_tensor = self.env_data_processing.batch_from_particles(belief=agent.belief, probabilities=self.bel_prob)
            self.hist_tensor = self.env_data_processing.cond_from_history(agent.history)
            new_belief, new_bel_prob = self.belief_net(belief_tensor.to(torch.float), self.hist_tensor)
            self.bel_prob = new_bel_prob

            new_belief, new_bel_prob = self.env_data_processing.particles_from_output(new_belief, new_bel_prob)
            return new_belief, new_bel_prob


    def updateNetworks(self, agent, reward, best_action_value):
        # QNet Loss 
        pred_q_values = self.q_net(self.hist_tensor)
        hist_conditioned_qvalues = self.hist_conditioned_qvalues
        # hist_conditioned_qvalues = torch.from_numpy(self.hist_conditioned_qvalues.copy()).to(torch.float)
        best_action_value = torch.tensor([best_action_value], requires_grad=True)
        assert pred_q_values.shape == hist_conditioned_qvalues.shape, "The shapes of Q(.|h) and Qnet(h) must match"

        # mask = ma.masked_where(hist_conditioned_qvalues==None, hist_conditioned_qvalues)
        # hist_conditioned_qvalues[mask.mask] = 0.
        # pred_q_values[mask.mask] = 0.

        mask = torch.isnan(hist_conditioned_qvalues)
        hist_conditioned_qvalues[mask] = 0.
        pred_q_values[mask] = 0.

        qnet_loss = F.mse_loss(pred_q_values.to(torch.float), hist_conditioned_qvalues)

        # Belief Net Loss
        # Assuming that the agent's history agent.history has been updated via agent.update_history() and a reward has been seen via env.state_transition()
        next_hist_tensor = self.env_data_processing.cond_from_history(agent.history)
        best_next_action = torch.max(self.q_net(next_hist_tensor))

        # delta = R + gamma*max(Q(hnext)) - Q(areal) 
        # Here R + gamma*max(Q(hnext)) is the return in bootstrap form. We are trying to change Q(areal) so that it is close to the return
        # Q(areal) = argmax( Q(.|h) ) = argmax( bel_prob * some_vector_from_sim ) = argmax( bel_net(old_bet, hist) * some_vector )
        # This means backward() will compute gradients for the bel_net 
        # requires_grad=False for bootstrapped_return since we do not want to update the Q_network in this step. We only want to backprop gradients to the belief network
        bootstrapped_return = reward + self.discount_factor*best_next_action
        bootstrapped_return = torch.tensor([bootstrapped_return], requires_grad=False) 

        belnet_loss = F.mse_loss(bootstrapped_return, best_action_value)

        # Update
        self.qnet_optim.zero_grad()
        qnet_loss.backward()
        self.qnet_optim.step()

        self.belnet_optim.zero_grad()
        belnet_loss.backward()
        self.belnet_optim.step()


        return qnet_loss, belnet_loss



