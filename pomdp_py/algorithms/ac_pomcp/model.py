import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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
        print(self.layers)

        modules = []
        for i in range(len(self.layers) - 1):        
            modules.append(nn.Linear(self.layers[i], self.layers[i+1]))
            if i != (len(self.layers) - 2):
                modules.append(nn.ReLU())

        self.encoder = nn.Sequential(*modules)

        print(self.encoder)


    def forward(self, x):
        torch.flatten(x)
        return self.encoder(x)


class FiLMEmbeddings(nn.Module):
    """
    Compute the FiLM Embedding from a conditioning input. https://arxiv.org/pdf/1709.07871.pdf

    Args:
        cond_dim (int): dimensionality of the conditioning vector (cond)
        cond (tensor): conditioning tensor
        x (tensor): tensor to which conditioning is applied
    """
    def __init__(self, cond_dim):
        super().__init__()
        self.cond_dim = cond_dim
        self.film_encoder = nn.Sequential(
          nn.Mish(),
          nn.Linear(self.cond_dim, 2),
        )

    def forward(self, x, cond):
        film_encoding = self.film_encoder(cond)
        scale = film_encoding[:,0]
        bias = film_encoding[:,1]

        return (scale*x.transpose(0,1) + bias).transpose(0,1)


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
    """

    def __init__(self, in_dim, latent_space_dim, cond_dim, out_dims, 
        encoder_hidden_layers, decoder_hidden_layers, output_prob_predicter_hidden_layers):
        super().__init__()
        self.encoder = LatentSpaceTf(in_dim, encoder_hidden_layers, latent_space_dim)
        self.film_encoder = FiLMEmbeddings(cond_dim)
        self.output_heads = []
        for i in out_dims:
            self.output_heads.append(LatentSpaceTf(latent_space_dim, decoder_hidden_layers[i], out_dims[i]))

        self.output_prob_predicter = LatentSpaceTf(latent_space_dim, output_prob_predicter_hidden_layers, batch_size)

    def forward(self, x, cond):
        encoded_input = self.encoder(x)
        conditioned_input = self.film_encoder(encoded_input, cond)
        outputs = []
        for output_head in self.output_heads:
            outputs.append(output_head(conditioned_input))


        # TODO: Apply constraints on the outputs. Softmax, upscaling etc.

        output_probabilities = self.output_prob_predicter(conditioned_input)

        return torch.cat(outputs, output_probabilities)

