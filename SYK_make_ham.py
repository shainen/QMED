import ed_build_hil as b
import numpy as np
#from scipy.sparse import lil_matrix, csr_matrix

#SITES = 16

#num_up_ferms = 4
#initial_fermion_up = [0,0,0,0,1,1,0,0,0,0]
#initial_fermion_up = [0,1,1,0]
#initial_fermion_up = [0 for _ in range(LENGTH)]
#initial_fermion_up[LENGTH//2] = 1
#for i in range(num_up_ferms//2): initial_fermion_up[LENGTH//2-1-i] = initial_fermion_up[LENGTH//2+i] = 1
initial_fermion_up = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
initial_state = b.State({'up': initial_fermion_up})

print(str(initial_state))

from math import factorial
SITES = len(initial_fermion_up)
num_up_ferms = sum(initial_fermion_up)
def fermion_comb(particles):
    return factorial(SITES)/factorial(SITES-particles)/factorial(particles)
DIMENSION = int(fermion_comb(num_up_ferms))

print(DIMENSION)

sigma = 1

J_real = np.random.normal(0,sigma,(SITES,SITES,SITES,SITES))
J_real = J_real - np.transpose(J_real,(1,0,2,3))
J_real = J_real - np.transpose(J_real,(0,1,3,2))
J_real = J_real + np.transpose(J_real,(2,3,0,1))

J_imag = np.random.normal(0,sigma,(SITES,SITES,SITES,SITES))
J_imag = J_imag - np.transpose(J_imag,(1,0,2,3))
J_imag = J_imag - np.transpose(J_imag,(0,1,3,2))
J_imag = J_imag - np.transpose(J_imag,(2,3,0,1))

J_coup = J_real + 1j*J_imag

print(J_coup)

def SYKint(state, state_dict, next_states, build_mat):
    return b.SYK_model(state, state_dict, next_states, build_mat, J_coup)

state_dict, [SYK_mat] = b.make_matrices_and_states(initial_state, DIMENSION, SYKint)

import pickle
pickle.dump( state_dict, open( "s"+str(SITES)+"SYK_statedict.p", "wb" ) )
b.save_sparse_csr("s"+str(SITES)+"_SYK_mat",SYK_mat)
np.save("J_coup",J_coup)
