import ed_build_hil as b
import numpy as np
#from scipy.sparse import lil_matrix, csr_matrix

LENGTH = 16

num_up_ferms = 2
#initial_fermion_up = [0,0,0,0,1,1,0,0,0,0]
#initial_fermion_up = [0,1,1,0]
initial_fermion_up = [0 for _ in range(LENGTH)]
#initial_fermion_up[LENGTH//2] = 1
for i in range(num_up_ferms//2): initial_fermion_up[LENGTH//2-1-i] = initial_fermion_up[LENGTH//2+i] = 1
initial_fermion_down = initial_fermion_up[:]
initial_state = b.State({'up': initial_fermion_up, 'down': initial_fermion_down})

print(str(initial_state))

from scipy.special import binom
def fermion_comb(particles): return binom(LENGTH, particles)
DIMENSION = int(fermion_comb(num_up_ferms) * fermion_comb(num_up_ferms))

print(DIMENSION)

state_dict, [hop_mat] = b.make_matrices_and_states(initial_state, DIMENSION, b.hop_lr_a1_ob)

import pickle
pickle.dump( state_dict, open( "s16_FH_lr_statedict.p", "wb" ) )
b.save_sparse_csr("l16_FH_lr_hop",hop_mat)

int_mat = b.make_diag_ops(state_dict, lambda x: np.dot(x['up'],x['down']))

b.save_sparse_csr("l16_FH_lr_int",int_mat)

