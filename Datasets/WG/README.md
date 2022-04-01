The folder contains the synthetic data generated from implicit numerical integration of the WG model in the principal space.

Data is normalized between -1.0 and 1.0.

Data is stored as mXn tensor where m is the number of samples and n is the number of features

The 'X' label contains the input data and the 'y' label contains the output data.

dstate.X and dstress.X are mX13 tensors containing void ratio(1), stress(3), plastic strain(3), strain(3) and strain increment(3).

dstate.y is a mX4 tensor containing the void ratio(1) and plastic strain increments(3), while dstress.y is a mX3 tensor containing the stress increments.

"dstate-04-plas.dat" and "dstress-04-plas.dat" include data from random loading paths with maximum strain magnitude of 4e-4 in
each loading step. Only the plastic steps are recorded.

"dstate-16-plas.dat" and "dstress-16-plas.dat" include data from random loading paths with maximum strain magnitude of 16e-4 in
each loading step. Only the plastic steps are recorded.
