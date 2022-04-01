""" Elasto-Plastic Neural Network (EPNN)

DEVELOPED AT:
                    COMPUTATIONAL GEOMECHANICS LABORATORY
                    DEPARTMENT OF CIVIL ENGINEERING
                    UNIVERSITY OF CALGARY, AB, CANADA
                    DIRECTOR: Prof. Richard Wan

DEVELOPED BY:
                    MAHDAD EGHBALIAN

MIT License

Copyright (c) 2022 Mahdad Eghbalian

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import epnn_module_utility as util
from torch import nn
import torch
import numpy as np


class Ann(nn.Module):
    def __init__(self, l_nodes, active_func, drop_p):
        super().__init__()
        ''' Builds a feedforward network with arbitrary layers

            Arguments
            ---------
            l_nodes: list containing number of nodes in each layer
            active_func: list containing activation functions used in each layer

        '''
        torch.manual_seed(10)

        # some network properties
        self.active_func = active_func
        self.l_nodes = l_nodes
        self.n_layers = len(self.l_nodes)
        self.n_inputs = self.l_nodes[0]
        self.n_outputs = self.l_nodes[-1]
        self.l_hidden = self.l_nodes[1:-1]
        self.n_hidden = len(self.l_hidden)
        self.drop_p = drop_p

        # initiate layers
        self.layers = nn.ModuleList([nn.Linear(l_nodes[0], l_nodes[1]).double()])

        # construct layers
        # regular
        # layer_sizes = zip(l_nodes[1:-1], l_nodes[2:])  # use tuple() to view
        # self.layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        # no bias for the output layer
        layer_sizes = zip(l_nodes[1:-2], l_nodes[2:-1])  # use tuple() to view
        self.layers.extend([nn.Linear(h1, h2).double() for h1, h2 in layer_sizes])
        self.layers.extend([nn.Linear(l_nodes[-2], l_nodes[-1], bias=False).double()])

        self.dropout = nn.Dropout(p=drop_p)

        # initialize weights (optional)
        # for ilayer in range(len(self.layers)):
        #     self.layers[ilayer].weight.data = torch.rand(self.l_nodes[ilayer + 1], self.l_nodes[ilayer])
        #     self.layers[ilayer].bias.data = torch.rand(self.layers[ilayer].weight.data.shape[0])

    def forward(self, x):
        """ Forward pass through the network """
        for ilayer in range(len(self.layers)):
            x = self.layers[ilayer](x)
            if ilayer != (len(self.layers) - 1):
                x = self.active_func[ilayer + 1](x)
            if ilayer != (len(self.layers) - 1):
                x = self.dropout(x)

        return x


class Data:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n_samples = self.x.shape[0]


class Solution:
    def __init__(self, strain, stress, strain_pl, vr):
        self.strain = np.transpose(strain)
        self.stress = np.transpose(stress)
        self.strain_pl = np.transpose(strain_pl)
        self.vr = [[vr]]
        self.ksecant = [[0.0]]
        self.p = [[util.mean(stress)]]
        self.q = [[util.eq(stress)]]
        strain_pl_j2 = util.j2(strain_pl)
        self.gamma_pl = [[np.sqrt((4.0 / 3.0) * strain_pl_j2)]]
        strain_j2 = util.j2(strain)
        self.gamma = [[np.sqrt((4.0 / 3.0) * strain_j2)]]

    def record(self, strain, stress, strain_pl, vr, ksecant):
        self.strain = np.append(self.strain, np.transpose(strain), axis=0)
        self.stress = np.append(self.stress, np.transpose(stress), axis=0)
        self.strain_pl = np.append(self.strain_pl, np.transpose(strain_pl), axis=0)
        self.vr = np.append(self.vr, [[vr]], axis=0)
        self.ksecant = np.append(self.ksecant, [[ksecant]], axis=0)

        p = util.mean(stress)
        q = util.eq(stress)
        strain_pl_j2 = util.j2(strain_pl)
        gamma_pl = np.sqrt((4.0 / 3.0) * strain_pl_j2)
        strain_j2 = util.j2(strain)
        gamma = np.sqrt((4.0 / 3.0) * strain_j2)

        self.p = np.append(self.p, [[p]], axis=0)
        self.q = np.append(self.q, [[q]], axis=0)
        self.gamma_pl = np.append(self.gamma_pl, [[gamma_pl]], axis=0)
        self.gamma = np.append(self.gamma, [[gamma]], axis=0)
