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
import epnn_classes as cls
import math
import torch
import time
import numpy as np


def datapp(data, train_p, cv_p, test_p):
    # shuffle the data
    torch.manual_seed(10)
    shuffled_indices = torch.randperm(data.x.shape[0])

    # choose main training data portion
    n_train_portion = train_p
    n_train = math.floor(n_train_portion * data.n_samples)
    data_train = cls.Data(x=data.x[shuffled_indices[0:n_train]],
                          y=data.y[shuffled_indices[0:n_train]])

    # choose cross-validation data portion
    n_cv_portion = cv_p
    n_cv = math.floor(n_cv_portion * data.n_samples)
    data_cv = cls.Data(x=data.x[shuffled_indices[n_train:(n_train + n_cv)]],
                       y=data.y[shuffled_indices[n_train:(n_train + n_cv)]])

    # choose test data portion
    n_test_portion = test_p
    n_test = math.floor(n_test_portion * data.n_samples)
    data_test = cls.Data(x=data.x[shuffled_indices[(n_train + n_cv):-1]],
                         y=data.y[shuffled_indices[(n_train + n_cv):-1]])

    return data_train, data_cv, data_test


def trainAnn(model11, model12, model2, data1, data2, datacv1, datacv2, criterion, optimizer11, optimizer12,
             optimizer2, epochs, datatest1, datatest2, ifsave, saveevery, ifplot, min1, min2, range1, range2,
             min3, range3, device, param, optimizer_param, nneurons, itrain, irepeat, nhlayers):

    # predictions pre-training
    train_error11, train_error12, train_error2 = util.pred_error(model11, model12, model2, data1, data2, criterion,
                                                                 min1, min2, range1, range2, min3, range3, device,
                                                                 param)
    cv_error11, cv_error12, cv_error2 = util.pred_error(model11, model12, model2, datacv1, datacv2, criterion,
                                                        min1, min2, range1, range2, min3, range3, device, param)
    test_error11, test_error12, test_error2 = util.pred_error(model11, model12, model2, datatest1, datatest2,
                                                              criterion, min1, min2, range1, range2, min3, range3,
                                                              device, param)

    print(f"errors on training data: {train_error11}, {train_error12}, {train_error2}")
    print(f"errors on cross-validation data: {cv_error11}, {cv_error12}, {cv_error2}")
    print(f"errors on test data: {test_error11}, {test_error12}, {test_error2}")

    print(f"gkratio: {param.item()}")

    start = time.time()
    model11.train()
    model12.train()
    model2.train()
    trends = np.zeros((0, 7))
    kounter = 0
    kounter3 = 0
    kounter2 = 0
    for e in range(epochs):
        optimizer11.zero_grad()
        optimizer12.zero_grad()
        optimizer2.zero_grad()
        optimizer_param.zero_grad()
        cost, cost11, cost12, cost2 = util.cost_function(model11, model12, model2, data1, data2, criterion,
                                                         min1, min2, range1, range2, min3, range3, device, param)
        cost.backward()
        optimizer11.step()
        optimizer12.step()
        optimizer2.step()
        optimizer_param.step()

        print(f"epoch: {e + 1}")
        print(f"Training losses: {cost.item(), cost11.item(), cost12.item(), cost2.item()}")

        if ifplot:
            train_error11, train_error12, train_error2 = util.pred_error(model11, model12, model2, data1, data2,
                                                                         criterion, min1, min2, range1, range2,
                                                                         min3, range3, device, param)
            cv_error11, cv_error12, cv_error2 = util.pred_error(model11, model12, model2, datacv1, datacv2, criterion,
                                                                min1, min2, range1, range2, min3, range3, device, param)
            cost_train = cost.item()

            trends = np.append(trends, np.array([[train_error11, train_error12, train_error2, cv_error11, cv_error12,
                                                  cv_error2, cost_train]]), 0)

        kounter += 1
        if ifsave and kounter == saveevery:
            checkpoint_state1 = {'ann_l_nodes': model11.l_nodes,
                                 'ann_active_func': model11.active_func,
                                 'ann_drop_p': model11.drop_p,
                                 'state_dict': model11.state_dict()}
            checkpoint_state2 = {'ann_l_nodes': model12.l_nodes,
                                 'ann_active_func': model12.active_func,
                                 'ann_drop_p': model12.drop_p,
                                 'state_dict': model12.state_dict()}
            checkpoint_stress = {'ann_l_nodes': model2.l_nodes,
                                 'ann_active_func': model2.active_func,
                                 'ann_drop_p': model2.drop_p,
                                 'state_dict': model2.state_dict(),
                                 'gkratio': param}
            torch.save(checkpoint_state1, 'checkpoint_state1_' + str(e+1) + '.pth')
            torch.save(checkpoint_state2, 'checkpoint_state2_' + str(e+1) + '.pth')
            torch.save(checkpoint_stress, 'checkpoint_stress_' + str(e+1) + '.pth')
            kounter = 0

        kounter3 += 1
        if ifplot and kounter3 == saveevery:
            util.data_dumper_dat(file_name='trends_train_' + str(e+1) + '_' + str(nneurons) + '_'
                                           + str(itrain) + '_' + str(irepeat) + '_' + str(nhlayers)
                                           + '.dat', outputset=trends)
            kounter3 = 0

        kounter2 += 1
        if kounter2 == 1000:  # report the runtime every 1000 epochs
            print(f"Total training run time so far: {(time.time() - start):.3f} seconds")
            kounter2 = 0

    # predictions after training
    train_error11, train_error12, train_error2 = util.pred_error(model11, model12, model2, data1, data2, criterion,
                                                                 min1, min2, range1, range2, min3, range3, device,
                                                                 param)
    cv_error11, cv_error12, cv_error2 = util.pred_error(model11, model12, model2, datacv1, datacv2, criterion,
                                                        min1, min2, range1, range2, min3, range3, device, param)
    test_error11, test_error12, test_error2 = util.pred_error(model11, model12, model2, datatest1, datatest2,
                                                              criterion, min1, min2, range1, range2, min3, range3,
                                                              device, param)

    print(f"errors on training data: {train_error11}, {train_error12}, {train_error2}")
    print(f"errors on cross-validation data: {cv_error11}, {cv_error12}, {cv_error2}")
    print(f"errors on test data: {test_error11}, {test_error12}, {test_error2}")

    print(f"gkratio: {param.item()}")

    return model11, model12, model2


def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    n_pre = cls.Ann(l_nodes=checkpoint['ann_l_nodes'], active_func=checkpoint['ann_active_func'],
                    drop_p=checkpoint['ann_drop_p'])

    n_pre.load_state_dict(checkpoint['state_dict'])

    return n_pre


def load_checkpoint2(filename):
    checkpoint = torch.load(filename)
    n_pre = cls.Ann(l_nodes=checkpoint['ann_l_nodes'], active_func=checkpoint['ann_active_func'],
                    drop_p=checkpoint['ann_drop_p'])

    n_pre.load_state_dict(checkpoint['state_dict'])

    param = checkpoint['gkratio']
    param = param.to('cpu')

    return n_pre, param
