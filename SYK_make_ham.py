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
initial_fermion_up = [1 for _ in range(10)] + [0 for _ in range(10)]
#initial_fermion_up = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
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

reflat=J_real.reshape(1,J_real.size)
imflat=J_imag.reshape(1,J_imag.size)
np.savetxt("flatJreal.CSV",reflat,delimiter=',')
np.savetxt("flatJimag.CSV",imflat,delimiter=',')

# datadir ='/projectnb/twambl/170207_1_s16f4/'   

# J_real = np.loadtxt(datadir+"flatJreal.CSV",delimiter=',')
# J_imag = np.loadtxt(datadir+"flatJimag.CSV",delimiter=',')
# J_coup = (J_real + 1j*J_imag).reshape(SITES,SITES,SITES,SITES)

#print(J_coup)

def SYKint(state, state_dict, next_states, build_mat):
    return b.SYK_model(state, state_dict, next_states, build_mat, J_coup)

funcs = [SYKint, b.correlation(1,2)]
#funcs = [SYKint] + [b.correlation(i,j) for i in range(SITES) for j in range(SITES)]

state_dict, mats = b.make_matrices_and_states(initial_state, DIMENSION, *funcs)

import pickle
pickle.dump( state_dict, open( "s"+str(SITES)+"SYK_statedict.p", "wb" ) )
b.save_sparse_csr("s"+str(SITES)+"_SYK_mat",mats[0])
# for i in range(SITES):
#     for j in range(SITES):
#         b.save_sparse_csr("s"+str(SITES)+"_corr"+str(i)+str(j),mats[SITES*i+j+1])

