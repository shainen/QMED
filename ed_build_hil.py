from scipy.sparse import lil_matrix
from copy import deepcopy
import numpy as np

class State(dict):
    def __str__(self):
        string = "{"
        for key, vals in sorted(self.items()):
            string = string + "'" + str(key) + "': " + str(vals) + ", "
        string = string[:-2] + "}"
        return string

def _add_to_mat(state, res_state, overlap, state_dict, next_states, mat):
    if res_state:
        if str(res_state) in state_dict:
            mat[state_dict[str(res_state)],state_dict[str(state)]] += overlap
        else:
            next_states.append(deepcopy(res_state))
            state_dict[str(res_state)] = len(state_dict)
            mat[state_dict[str(res_state)],state_dict[str(state)]] += overlap

def make_matrices_and_states(initial_state, dim, *funcs):
    mats = [lil_matrix((dim,dim)) for _ in funcs]
    state_dict = {str(initial_state): 0}
    next_states = [initial_state]
    while next_states:
        current_state = next_states.pop()
        for ind, func in enumerate(funcs):
            func(current_state, state_dict, next_states, mats[ind])
    csr_mats = [mat.tocsr() for mat in mats]
    #check that the dimension of the Hilbert space is what we predicted
    if dim==len(state_dict):
        print('Dimension of Hilbert space is what was guessed')
    else:
        print('guessed: ' + str(dim) + ' actual: ' + str(len(state_dict)))
    return (state_dict, csr_mats)

def _fermion_hop_1d(state, i, direc, spin, alpha=1):
    length = len(state[spin])
    if state[spin][i] == 0 and state[spin][(i+direc)%length] == 1:
        res_state = deepcopy(state)
        res_state[spin][i] = 1
        res_state[spin][(i+direc)%length] = 0
        num_between = (sum(state[spin][i+1:(i+direc)%length]) 
            if direc > 0 else sum(state[spin][(i+direc)%length+1:i]))
        return (res_state, (-1)**num_between/abs(direc)**alpha)
    else:
        return (None, None)

def hop_nn_ob(state, state_dict, next_states, build_hop):
    length = len(state['up'])
    for spin in ('up','down'):
        for i in range(length-1):
            res_state, overlap = _fermion_hop_1d(state,i,1,spin)
            _add_to_mat(state, res_state, overlap, state_dict, next_states, build_hop)
        for i in range(1,length):
            res_state, overlap = _fermion_hop_1d(state,i,-1,spin)
            _add_to_mat(state, res_state, overlap, state_dict, next_states, build_hop)

def hop_lr_a1_ob(state, state_dict, next_states, build_hop):
    length = len(state['up'])
    for spin in ('up','down'):
        for i in range(length):
            for direc in range(-i,length-i):
                res_state, overlap = _fermion_hop_1d(state,i,direc,spin,1)
                _add_to_mat(state, res_state, overlap, state_dict, next_states, build_hop)

def make_diag_ops(state_dict, func):
    dim = len(state_dict)
    mat = lil_matrix((dim,dim))
    for ind, val in state_dict.items():
        mat[val,val] = func(eval(ind))
    csr_mat = mat.tocsr()
    return csr_mat

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
