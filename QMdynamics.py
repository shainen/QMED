import ed_build_hil as b
import numpy as np
import pickle

data_dir="/data/shainen/170109_9_s16_FH2d_nn_make_mats/"
#data_dir=""

hop_mat=b.load_sparse_csr(data_dir+"l16_FH2d_nn_hop.npz")
int_mat=b.load_sparse_csr(data_dir+"l16_FH2d_nn_int.npz")
state_dict = pickle.load( open( data_dir+"s16_FH2d_nn_statedict.p", "rb" ) )

U = 0

LENGTH = 16
DIMENSION = hop_mat.shape[0]

TMAX = 10
TSTEPS = 100
DT = TMAX/TSTEPS
times = np.linspace(0,TMAX,TSTEPS,endpoint=False)

num_up_diag = np.zeros((DIMENSION, LENGTH))
for key, val in state_dict.items():
    num_up_diag[val] = np.array(eval(key)['up'][:])
num_up_diag = num_up_diag.transpose()

ham = - hop_mat + U*int_mat

hop_mat = None
int_mat = None
state_dict = None

init_vec = np.zeros(DIMENSION)
init_vec[0] = 1

num_up_t = b.diag_ops_dynamics(init_vec, ham, TSTEPS, DT, num_up_diag)

np.savetxt("s"+str(LENGTH)+"_FH2d_nn_u"+str(U)+".dat",num_up_t)
