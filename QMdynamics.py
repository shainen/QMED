import ed_build_hil as b
import numpy as np
import pickle

data_dir="/projectnb/twambl/170525_1_s18f9_makemats/"
data_dir2="/projectnb/twambl/170527_1_s18f9_makepert/"
LENGTH = 18

SYK_ham=b.load_sparse_csr(data_dir+"s"+str(LENGTH)+"_SYK_mat.npz")
pert_ham=b.load_sparse_csr(data_dir2+"s"+str(LENGTH)+"_pert_mat.npz")

state_dict = pickle.load( open( data_dir+"s"+str(LENGTH)+"SYK_statedict.p", "rb" ) )

#U = 2

#ham = - hop_mat + U*int_mat

DIMENSION = len(state_dict)

from constants import *

TMAX = 0.06
#TPERT = 0.05
EXTRA = 0.25

DT = 0.0001

TSTEPS = int(TMAX/DT)
TSTEPS_PERT = int(TPERT/DT)
TSTEPS_BACK = int((1+EXTRA)*TMAX/DT)

#times = np.linspace(0,TMAX,TSTEPS,endpoint=True)

num_up_diag = np.zeros((DIMENSION, LENGTH))
for key, val in state_dict.items():
    num_up_diag[val] = np.array(eval(key)['up'][:])
num_up_diag = num_up_diag.transpose()

# hop_mat = None
# int_mat = None
# state_dict = None

init_vec = np.zeros(DIMENSION)
init_vec[0] = 1

num_up_t_forward, psi_t = b.diag_ops_dynamics(init_vec, SYK_ham, TSTEPS, DT, num_up_diag)
num_up_t_pert, psi_p = b.diag_ops_dynamics(psi_t, pert_ham, TSTEPS_PERT, DT, num_up_diag)
# overlap = np.vdot(psi_t,psi_p)
# norm = np.sqrt(1-np.absolute(overlap)**2)
# phi = (psi_p - overlap*psi_t)/norm
# print "\noverlap: " + str(overlap)
# print "\namount of preperp: " + str(np.vdot(psi_t,phi))
# print "\namount of postperp: " + str(np.vdot(psi_p,phi))
# print "\nnorm of new vector: " + str(np.vdot(phi,phi))
num_up_t_backwards, psi_f = b.diag_ops_dynamics(phi_p, SYK_ham, TSTEPS_BACK, -DT, num_up_diag)
#num_up_t , exp_corrs = b.both_ops_dynamics(init_vec, ham, TSTEPS, DT, num_up_diag, corr_mats)

#np.savetxt("s"+str(LENGTH)+"_FH_lr_u"+str(U)+".dat",num_up_t)
np.savetxt("s"+str(LENGTH)+"_SYKf.dat",num_up_t_forward)
np.savetxt("s"+str(LENGTH)+"_SYKp.dat",num_up_t_pert)
np.savetxt("s"+str(LENGTH)+"_SYKb.dat",num_up_t_backwards)
#np.savetxt("s"+str(LENGTH)+"_re_corrs.dat",np.real(exp_corrs))
#np.savetxt("s"+str(LENGTH)+"_im_corrs.dat",np.imag(exp_corrs))
