import ed_build_hil as b
import numpy as np
import pickle

data_dir="/projectnb/twambl/170216_1_s9f6/"
#data_dir="/Users/shainen/Dropbox/Research/fTWA/fermion velocity/QMED/"

LENGTH = 9

ham=b.load_sparse_csr(data_dir+"s"+str(LENGTH)+"_SYK_mat.npz")
corr_mats = [b.load_sparse_csr(data_dir+"s"+str(LENGTH)+"_corr"+str(i)+str(j)+".npz") for i in range(LENGTH) for j in range(LENGTH)]
#int_mat=b.load_sparse_csr(data_dir+"l16_FH_nn_int.npz")
state_dict = pickle.load( open( data_dir+"s"+str(LENGTH)+"SYK_statedict.p", "rb" ) )

#U = 0


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

#num_up_t = b.diag_ops_dynamics(init_vec, ham, TSTEPS, DT, num_up_diag)
num_up_t , exp_corrs = b.both_ops_dynamics(init_vec, ham, TSTEPS, DT, num_up_diag, corr_mats)

np.savetxt("s"+str(LENGTH)+"_SYK.dat",num_up_t)
np.savetxt("s"+str(LENGTH)+"_re_corrs.dat",np.real(exp_corrs))
np.savetxt("s"+str(LENGTH)+"_im_corrs.dat",np.imag(exp_corrs))
