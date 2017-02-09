import ed_build_hil as b
import numpy as np
import pickle

data_dir="/Users/shainen/Dropbox/Research/fTWA/SYK model/data/170207_1_s16f4/"

ham=b.load_sparse_csr(data_dir+"s16_SYK_mat.npz")
#int_mat=b.load_sparse_csr(data_dir+"l16_FH_nn_int.npz")
state_dict = pickle.load( open( data_dir+"s16SYK_statedict.p", "rb" ) )

#U = 0

LENGTH = 16
DIMENSION = ham.shape[0]

TMAX = 0.01
TSTEPS = 100
DT = TMAX/TSTEPS
times = np.linspace(0,TMAX,TSTEPS,endpoint=False)

num_up_diag = np.zeros((DIMENSION, LENGTH))
for key, val in state_dict.items():
    num_up_diag[val] = np.array(eval(key)['up'][:])
num_up_diag = num_up_diag.transpose()

#ham = - hop_mat + U*int_mat

#hop_mat = None
#int_mat = None
state_dict = None

init_vec = np.zeros(DIMENSION)
init_vec[0] = 1

num_up_t = b.diag_ops_dynamics(init_vec, ham, TSTEPS, DT, num_up_diag)

np.savetxt("s"+str(LENGTH)+"_SYK.dat",num_up_t)
