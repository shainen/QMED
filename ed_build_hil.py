import time
from scipy.sparse import lil_matrix, csr_matrix
from copy import deepcopy
import numpy as np
from scipy.sparse.linalg import expm_multiply

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
    mats = [lil_matrix((dim,dim), dtype=np.cfloat) for _ in funcs]
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

def _coord_from_num(num,dim):
    coord = []
    num_left = num
    div = np.product(dim[1:])
    for base in dim[1:]:
        digit = num_left//div
        coord.append(digit)
        num_left -= digit * div
        div //= base
    coord.append(num_left)
    return np.array(coord)

def _num_from_coord(coord,dim):
    num = coord[-1]
    base = 1
    for d, c in zip(dim[:0:-1], coord[-2::-1]):
        base *= d
        num += base * c
    return num

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

def _fermion_hop(state, i, direc, spin, alpha=1, dim=False):
    length = len(state[spin])
    if not dim:
        dim = [length]
    if state[spin][i] == 0 and state[spin][(i+direc)%length] == 1:
        res_state = deepcopy(state)
        res_state[spin][i] = 1
        res_state[spin][(i+direc)%length] = 0
        num_between = (sum(state[spin][i+1:(i+direc)%length]) 
            if direc > 0 else sum(state[spin][(i+direc)%length+1:i]))
        distance = np.linalg.norm(_coord_from_num(i,dim)
                                  -_coord_from_num((i+direc)%length,dim))
        return (res_state, (-1)**num_between/distance**alpha)
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

def hop_nn_2d_ob(state, state_dict, next_states, build_hop, dim):
    sites = len(state['up'])
    for spin in ('up','down'):
        for i in range(sites):
            for vec in np.array([[-1,0],[1,0],[0,-1],[0,1]]):
                new_coord = _coord_from_num(i,dim)+vec
                if np.less_equal([0,0],new_coord).all() and np.less(new_coord,dim).all():
                    res_state, overlap = _fermion_hop(state,i,_num_from_coord(new_coord,dim)-i,spin,1,dim)
                    _add_to_mat(state, res_state, overlap, state_dict, next_states, build_hop)

def hop_lr_a1_ob(state, state_dict, next_states, build_hop):
    length = len(state['up'])
    for spin in ('up','down'):
        for i in range(length):
            for direc in range(-i,length-i):
                res_state, overlap = _fermion_hop_1d(state,i,direc,spin,1)
                _add_to_mat(state, res_state, overlap, state_dict, next_states, build_hop)

def hop_lr_ob(state, state_dict, next_states, build_hop,alpha,dim):
    length = len(state['up'])
    for spin in ('up','down'):
        for i in range(length):
            for direc in range(-i,length-i):
                res_state, overlap = _fermion_hop(state,i,direc,spin,alpha,dim)
                _add_to_mat(state, res_state, overlap, state_dict, next_states, build_hop)

def SYK_model(state, state_dict, next_states, build_mat, J_coup):
    sites = len(state['up'])
    for i in range(sites):
        for j in range(sites):
            for k in range(sites):
                for l in range(sites):
                    res_state, overlap = _fermion_hop(state,j,l-j,'up',0)
                    if res_state:
                        res_state2, overlap2 = _fermion_hop(res_state,i,k-i,'up',0)
                        if res_state2:
                            _add_to_mat(state, res_state2, -overlap*overlap2*J_coup[i,j,k,l], state_dict, next_states, build_mat)
                    if k==j:
                        res_state, overlap = _fermion_hop(state,i,l-i,'up',0)
                        if res_state:
                            _add_to_mat(state, res_state, overlap*J_coup[i,j,k,l], state_dict, next_states, build_mat)
                
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

def _expec_diag_ops(psi,ops):
    return np.real(list(
                map(lambda op: np.einsum('i,i,i',np.conj(psi),op,psi),
                   ops)
            ))

def diag_ops_dynamics(psi_0, ham, tsteps, dt, ops):
    ops_t = []
    psi_t = psi_0

    ops_t.append(_expec_diag_ops(psi_t,ops))
    for _ in range(tsteps - 1):
        t1 = time.time()
        psi_t = expm_multiply(1j*dt*ham,psi_t)
        t2 = time.time()
        print(t2-t1)
        ops_t.append(_expec_diag_ops(psi_t,ops))
        t3 = time.time()
        print(t3-t2)

    return np.array(ops_t).transpose()
