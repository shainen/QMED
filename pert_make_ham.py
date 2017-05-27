import ed_build_hil as b
import numpy as np
import pickle

data_dir="/projectnb/twambl/170525_1_s18f9_makemats/"
LENGTH = 18
state_dict = pickle.load( open( data_dir+"s"+str(LENGTH)+"SYK_statedict.p", "rb" ) )

sigma = 1

rands = np.random.normal(0,sigma,(SITES,SITES))
T_coup = (rands + np.transpose(rands,(1,0)))/np.sqrt(2)

T_flat = T_coup.reshape(1,T_coup.size)
np.savetxt("flatTcoup.CSV",T_flat,delimiter=',')

def pert_ham_wconst(state, state_dict, next_states, build_mat):
    return b.quad_pert_ham(state, state_dict, next_states, build_mat, T_coup)

funcs = [pert_ham_wconst]
#funcs = [SYKint] + [b.correlation(i,j) for i in range(SITES) for j in range(SITES)]

state_dict_back, mats = b.make_matrices_and_states(state_dict, DIMENSION, *funcs)

#import pickle
#pickle.dump( state_dict, open( "s"+str(SITES)+"SYK_statedict.p", "wb" ) )
b.save_sparse_csr("s"+str(SITES)+"_pert_mat",mats[0])
# for i in range(SITES):
#     for j in range(SITES):
#         b.save_sparse_csr("s"+str(SITES)+"_corr"+str(i)+str(j),mats[SITES*i+j+1])

