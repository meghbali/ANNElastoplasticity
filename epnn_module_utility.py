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

import torch
import pickle
import numpy as np


def data_loader_pt(file_name):
    return torch.load(file_name)


def data_loader_dat(file_name):
    get_data = pickle.load(open(file_name, "rb"))
    return get_data


def data_dumper_dat(file_name, outputset):
    pickle.dump(outputset, open(file_name, "wb"))
    return


def pred_error(n11, n12, n2, data1, data2, criterion, min1, min2, range1, range2, min3, range3, device, param):
    n11.eval()
    n12.eval()
    n2.eval()

    # if normalized in [-1, 1]
    coeff1 = 2.0
    coeff2 = 1.0

    # if normalized in [0, 1]
    # coeff1 = 1.0
    # coeff2 = 0.0

    input11 = n11.forward(data1.x)
    input12 = n12.forward(data1.x)
    target11 = data1.y[:, 0:1]
    target12 = data1.y[:, 1:4]
    out11 = 100.0 * (torch.norm(input11 - target11) / torch.norm(target11))
    out12 = 100.0 * (torch.norm(input12 - target12) / torch.norm(target12))

    input1m = torch.cat((input11, data1.x[:, [0, 1, 2, 3, 7, 8, 9, 10, 11, 12]]), 1)

    input21 = n2.forward(input1m)

    # constant tensors
    # oneten1 = torch.ones(3, 1, dtype=torch.float32)
    oneten1 = torch.ones(3, 1).double()
    oneten1 = oneten1.to(device)
    # oneten2 = torch.ones(data2.y.shape[0], data2.y.shape[1], dtype=torch.float32)
    oneten2 = torch.ones(data2.y.shape[0], data2.y.shape[1]).double()
    oneten2 = oneten2.to(device)

    dstrain = data1.x[:, 10:]
    dstrain_real = torch.mul(dstrain + coeff2, range3) / coeff1 + min3
    # dstrainpl = input12  # predicted plastic strain increment
    dstrainpl = target12  # actual plastic strain increment
    dstrainpl_real = torch.mul(dstrainpl + coeff2, range1[1:4]) / coeff1 + min1[1:4]
    dstrainel = dstrain_real - dstrainpl_real
    dstrainelv = torch.matmul(dstrainel, oneten1)
    dstrainelvten = torch.mul(dstrainelv, oneten2)

    mu = torch.mul(param, input21[:, 0:1])

    input22 = 2.0 * torch.mul(mu, dstrainel)
    input23 = torch.mul((input21[:, 0:1] - (2.0 / 3.0) * mu), dstrainelvten)
    input24 = input22 + input23
    input2 = coeff1 * torch.div((input24 - min2), range2) - coeff2

    target2 = data2.y
    out2 = 100.0 * (torch.norm(input2 - target2) / torch.norm(target2))

    n11.train()
    n12.train()
    n2.train()
    return out11.item(), out12.item(), out2.item()


def cost_function(n11, n12, n2, data1, data2, criterion, min1, min2, range1, range2, min3, range3, device, param):
    # if normalized in [-1, 1]
    coeff1 = 2.0
    coeff2 = 1.0

    # if normalized in [0, 1]
    # coeff1 = 1.0
    # coeff2 = 0.0

    input11 = n11.forward(data1.x)
    input12 = n12.forward(data1.x)
    target11 = data1.y[:, 0:1]
    target12 = data1.y[:, 1:4]
    cost11 = criterion(input11, target11)
    cost12 = criterion(input12, target12)

    input1m = torch.cat((input11, data1.x[:, [0, 1, 2, 3, 7, 8, 9, 10, 11, 12]]), 1)

    input21 = n2.forward(input1m)

    # constant tensors
    # oneten1 = torch.ones(3, 1, dtype=torch.float32)
    oneten1 = torch.ones(3, 1).double()
    oneten1 = oneten1.to(device)
    # oneten2 = torch.ones(data2.y.shape[0], data2.y.shape[1], dtype=torch.float32)
    oneten2 = torch.ones(data2.y.shape[0], data2.y.shape[1]).double()
    oneten2 = oneten2.to(device)

    dstrain = data1.x[:, 10:]
    dstrain_real = torch.mul(dstrain + coeff2, range3) / coeff1 + min3
    # dstrainpl = input12  # predicted plastic strain increment
    dstrainpl = target12  # actual plastic strain increment
    dstrainpl_real = torch.mul(dstrainpl + coeff2, range1[1:4]) / coeff1 + min1[1:4]
    dstrainel = dstrain_real - dstrainpl_real
    dstrainelv = torch.matmul(dstrainel, oneten1)
    dstrainelvten = torch.mul(dstrainelv, oneten2)

    mu = torch.mul(param, input21[:, 0:1])

    input22 = 2.0 * torch.mul(mu, dstrainel)
    input23 = torch.mul((input21[:, 0:1] - (2.0 / 3.0) * mu), dstrainelvten)
    input24 = input22 + input23
    input2 = coeff1 * torch.div((input24 - min2), range2) - coeff2

    target2 = data2.y
    cost2 = criterion(input2, target2)

    cost = cost11 + cost12 + cost2
    return cost, cost11, cost12, cost2


def mean(inp):
    output = (inp[0][0] + inp[1][0] + inp[2][0]) / 3.0
    return output


def deviator(inp):
    p = mean(inp)
    s = np.array([[inp[0][0] - p], [inp[1][0] - p], [inp[2][0] - p], [inp[3][0]], [inp[4][0]], [inp[5][0]]])
    return s


def j2(inp):
    s = deviator(inp)
    out = 0.0
    for i in range(6):
        out += s[i][0] ** 2.0 / 2.0
    return out


def eq(inp):
    j2_val = j2(inp)
    out = np.sqrt(3.0 * j2_val)
    return out
