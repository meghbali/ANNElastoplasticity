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

# ==================== import modules
import epnn_module_main as main
import epnn_classes as cls
import torch
from torch import nn, optim


def mainfunc(nneurons,  itrain, irepeat, device, data_train_state, data_train_stress, data_cv_state, data_cv_stress,
             data_test_state, data_test_stress, min_stress, min_state, range_state, range_stress, min_dstrain,
             range_dstrain, nhlayers1):

    # ========= create EPNN
    nhlayers2 = 4  # for sub-networks with different number of layers than the input
    nneurons2 = 75  # for sub-networks with different number of neurons than the input
    ann_hl_nodes_1 = [nneurons] * nhlayers1
    ann_hl_nodes_2 = [nneurons2] * nhlayers2

    ann_l_nodes_state1 = [data_train_state.x.shape[1]]
    ann_l_nodes_state2 = [data_train_state.x.shape[1]]
    ann_l_nodes_stress = [data_train_stress.x.shape[1] + data_train_state.y.shape[1] - 6]

    ann_l_nodes_state1.extend(ann_hl_nodes_1)
    ann_l_nodes_state2.extend(ann_hl_nodes_2)
    ann_l_nodes_stress.extend(ann_hl_nodes_1)

    ann_l_nodes_state1.extend([data_train_state.y.shape[1] - 3])
    ann_l_nodes_state2.extend([data_train_state.y.shape[1] - 1])
    ann_l_nodes_stress.extend([1])

    # choose the activation function
    # nn.Sigmoid(), nn.Tanh(), nn.LogSigmoid(), nn.ReLU(), nn.LeakyReLU(), nn.ELU(), nn.Softmax(dim=1),
    # nn.LogSoftmax(dim=1), nn.Linear()
    active_func = nn.LeakyReLU()
    ann_active_func1 = nn.ModuleList([active_func])
    ann_active_func2 = nn.ModuleList([active_func])
    ann_active_func1.extend([active_func for ii in range(nhlayers1)])
    ann_active_func2.extend([active_func for ii in range(nhlayers2)])
    ann_active_func1.extend([active_func])
    ann_active_func2.extend([active_func])

    # use dropout only for hidden layers. 0.0 means no dropout
    ann_drop_p = 0.0

    # create ANN object
    n1_state1 = cls.Ann(l_nodes=ann_l_nodes_state1, active_func=ann_active_func1, drop_p=ann_drop_p)
    n1_state2 = cls.Ann(l_nodes=ann_l_nodes_state2, active_func=ann_active_func2, drop_p=ann_drop_p)
    n1_stress = cls.Ann(l_nodes=ann_l_nodes_stress, active_func=ann_active_func1, drop_p=ann_drop_p)
    n1_state1.to(device)
    n1_state2.to(device)
    n1_stress.to(device)
    # next(n1.parameters()).is_cuda  # to check if the network is on GPU

    # choose the loss function
    # L1Loss (mean absolute error), MSELoss (mean squared error)
    criterion = nn.L1Loss()

    # the network constant to be optimized
    gkratio = torch.tensor([[0.45]])
    gkratio = gkratio.to(device)
    gkratio.requires_grad = True

    # choose the optimizer
    # optim.Adam() optim.AdamW() optim.SGD() optim.LBFGS()
    optimizer_stress = optim.Adam(n1_stress.parameters(), lr=0.01)
    optimizer_state1 = optim.Adam(n1_state1.parameters(), lr=0.001)
    optimizer_state2 = optim.Adam(n1_state2.parameters(), lr=0.001)
    optimizer_ratio = optim.Adam([gkratio], lr=0.001)

    # ==================== train EPNN
    if_train_plot = True
    if_train_save = True

    nepochs_train = 10000  # number of training epochs
    save_every = 10000  # save the outputs in every save_every epochs

    n1_state1, n1_state2, n1_stress = main.trainAnn(model11=n1_state1, model12=n1_state2, model2=n1_stress,
                                                    data1=data_train_state, data2=data_train_stress,
                                                    datacv1=data_cv_state, datacv2=data_cv_stress,
                                                    criterion=criterion, optimizer11=optimizer_state1,
                                                    optimizer12=optimizer_state2, optimizer2=optimizer_stress,
                                                    epochs=nepochs_train, datatest1=data_test_state,
                                                    datatest2=data_test_stress, ifsave=if_train_save,
                                                    saveevery=save_every, ifplot=if_train_plot, min1=min_state,
                                                    min2=min_stress, range1=range_state, range2=range_stress,
                                                    min3=min_dstrain, range3=range_dstrain, device=device,
                                                    param=gkratio, optimizer_param=optimizer_ratio, nneurons=nneurons,
                                                    itrain=itrain, irepeat=irepeat, nhlayers=nhlayers1)
