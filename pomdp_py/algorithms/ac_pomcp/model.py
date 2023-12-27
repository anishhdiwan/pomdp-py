import numpy as np
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




# ### TESTING ###
# n, k = 5, 5
# num_particles = 6
# hist_dim = 10


# from rocksample_data_utils import RocksampleDataProcessing
# from pomdp_problems.rocksample import rocksample_problem as rs
# init_state, rock_locs = rs.RockSampleProblem.generate_instance(n, k)

# belief_type = "uniform"
# init_belief = rs.init_particles_belief(k, num_particles, init_state, belief=belief_type)
# # print(init_belief)

# data_processing = RocksampleDataProcessing(n=n, k=k, t=hist_dim, bel_size=num_particles)


# belief_tensor = data_processing.batch_from_particles(belief=init_belief, probabilities=np.full((num_particles), float(1/num_particles)))
# cond_tensor = data_processing.cond_from_history(history=())

# # print("-----")
# # print(f"Belief Tensor {num_particles} x [posx posy | {k} x rocktype | terminal+state | prob]")
# # print(belief_tensor)
# # print("Conditioning [a, o, a, o ..]")
# # print(cond_tensor)

# belief_net = MultiHeadAutoencoder(in_dim=k+4, latent_space_dim=128, cond_dim=hist_dim, out_dims=[2, k, 1], 
#         encoder_hidden_layers=[64], decoder_hidden_layers=[[64, 32], [64], [64, 32, 8]], 
#         output_prob_predicter_hidden_layers=[256, 126, 64, 8], batch_size=num_particles, n=n)

# new_belief, new_probabilities = belief_net(belief_tensor, cond_tensor)


# # print("-----")
# # print(f"Model Output {num_particles} x [posx posy | {k} x rocktype | terminal+state]")
# # print(new_belief)
# # print("Probabilities")
# # print(new_probabilities)


# new_belief, new_probabilities = data_processing.particles_from_output(new_belief, new_probabilities)
# # print("-----")
# print(new_belief)
# print(new_probabilities)


# # q_net = dueling_net(n_actions=data_processing.num_actions, in_dim=hist_dim, encoder_hidden_layers=[64], latent_space_dim=128, 
# #         state_decoder_hidden_layers=[64, 32, 8], advantage_decoder_hidden_layers=[64, 32])

# # q_values = q_net(cond_tensor)

# # print("Q(.|h)")
# # print(q_values)




