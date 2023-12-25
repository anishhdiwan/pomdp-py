import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

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


        # Take in the whole batch of conditioned inputs in the latent space and compute new probabilities
        self.output_prob_predicter = nn.Sequential(
            LatentSpaceTf(batch_size*latent_space_dim, output_prob_predicter_hidden_layers, batch_size),
            nn.Softmax(dim=0)
            )


    def forward(self, x, cond):
        encoded_input = self.encoder(x)
        conditioned_input = self.film_encoder(encoded_input, cond)

        outputs = []
        for output_head in self.output_heads:
            outputs.append(output_head(conditioned_input))

        output_probabilities = self.output_prob_predicter(conditioned_input.flatten())
        concatenated_output = torch.cat((outputs), 1).to(torch.float)

        return concatenated_output, output_probabilities




### TESTING ###
def temp_generate_sample(n,k):
    test_sample = torch.zeros((k+4))
    test_sample[:2] = torch.randint(0, n-1, (2,))

    mask = torch.rand(k+2)
    mask[:-1] = mask[:-1] < 0.5
    test_sample[2:] = mask

    return test_sample

def temp_generate_cond():
    test_cond = torch.zeros((2*3))
    for i in range(len(test_cond)):
        if (i+1)%2 != 0:
            test_cond[i] = torch.randint(0, 6, (1,))
        else:
            test_cond[i] = torch.rand(1)
    return test_cond

n, k = 5, 5
b_size = 4
device = 'cuda'
batch = torch.zeros(b_size,k+4)
for i in range(len(batch)):
    batch[i] = temp_generate_sample(n,  k)

cond = temp_generate_cond()


batch.to(torch.device(device)).to(torch.float)
cond.to(torch.device(device)).to(torch.float)

print(f"Batch {b_size} x [posx posy | {k} x rocktype | terminal+state | prob]")
print(batch)
print("Cond [a, o, a, o ..]")
print(cond)

belief_net = MultiHeadAutoencoder(k+4, 128, 6, [2, k, 1], [64], [[64, 32], [64], [64, 32, 8]], [256, 126, 64, 8], b_size, n)
out, prob = belief_net(batch, cond)

print(out)
print(prob)

