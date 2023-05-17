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

# import required modules
import epnn_main as main
import epnn_module_main as mmain
import epnn_classes as cls
import epnn_module_utility as util
import torch
import numpy as np


# ========= set device
if_gpu = True  # True=gpu, False=cpu
device = "cpu"
if if_gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= load data to device (refer to the readme file for dataset for details)
# load state data
get_data = util.data_loader_dat(file_name='Datasets/WG/dstate-16-plas.dat')
XX = torch.from_numpy(get_data['X'])
yy = torch.from_numpy(get_data['y'])
X, y = XX.to(device), yy.to(device)

# create data object for state variables
data_state = cls.Data(x=X, y=y)

# extract additional variables from state data
min_state = torch.from_numpy(get_data['miny'])
range_state = torch.from_numpy(get_data['rangey'])
min_dstrain = torch.from_numpy(get_data['minx'][10:])
range_dstrain = torch.from_numpy(get_data['rangex'][10:])
min_state, range_state, min_dstrain, range_dstrain = min_state.to(device), range_state.to(device), \
                                                     min_dstrain.to(device),  range_dstrain.to(device)

# load stress data
get_data = util.data_loader_dat(file_name='Datasets/WG/dstress-16-plas.dat')
XX = torch.from_numpy(get_data['X'])
yy = torch.from_numpy(get_data['y'])
X, y = XX.to(device), yy.to(device)

# create data object for stress
data_stress = cls.Data(x=X, y=y)

# extract additional variables from stress data
min_stress = torch.from_numpy(get_data['miny'])
range_stress = torch.from_numpy(get_data['rangey'])
min_stress, range_stress = min_stress.to(device), range_stress.to(device)

# ========= create training, cross-validation and test sets from data
data_train_state1, data_cv_state, data_test_state = mmain.datapp(data=data_state, train_p=0.6, cv_p=0.2, test_p=0.2)
data_train_stress1, data_cv_stress, data_test_stress = mmain.datapp(data=data_stress, train_p=0.6, cv_p=0.2, test_p=0.2)

# ===  main
# creat an array containing different training set size (learning curve)
ntrain_size = 40  # number of different training set sizes
train_size_log10 = np.linspace(1, np.log10(data_train_state1.x.shape[0]), num=ntrain_size)
train_size_float = 10 ** train_size_log10
train_size = train_size_float.astype(int)

for ihlayers in range(3, 4):  # loop over number of layers
    for ineurons in range(60, 61):  # loop over number of neurons
        train_kounter = ntrain_size - 1
        for itrain in train_size[train_kounter:ntrain_size]:
            train_kounter += 1
            for irepeat in range(0, 1):  # shuffle data and train again
                torch.manual_seed(10 + irepeat)
                shuffled_indices = torch.randperm(data_train_state1.x.shape[0])
                data_train_state = cls.Data(x=data_train_state1.x[shuffled_indices[0:itrain]],
                                            y=data_train_state1.y[shuffled_indices[0:itrain]])
                data_train_stress = cls.Data(x=data_train_stress1.x[shuffled_indices[0:itrain]],
                                             y=data_train_stress1.y[shuffled_indices[0:itrain]])

                # data_train_state = data_train_state1
                # data_train_stress = data_train_stress1

                main.mainfunc(ineurons, train_kounter, irepeat, device, data_train_state, data_train_stress,
                              data_cv_state, data_cv_stress, data_test_state, data_test_stress, min_stress,
                              min_state, range_state, range_stress, min_dstrain, range_dstrain, ihlayers)
