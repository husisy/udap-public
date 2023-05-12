import os
import json
import itertools
import functools
import scipy.sparse
import numpy as np
import torch

hf_kron = lambda x: functools.reduce(np.kron, x)

def save_index_to_file(file, key_str=None, index=None):
    if os.path.exists(file):
        with open(file, 'r', encoding='utf-8') as fid:
            all_data = json.load(fid)
    else:
        all_data = dict()
    if (index is not None) and len(index)>0:
        if isinstance(index[0], int): #[2,3,4]
            index_batch = [[int(x) for x in index]]
        elif isinstance(index[0], str): #["2 3 4"]
            index_batch = [[int(y) for y in x.split(' ')] for x in index]
        else: #[[2,3,4]]
            index_batch = [[int(y) for y in x] for x in index]
        data_i = [[int(y) for y in x.split(' ')] for x in all_data.get(key_str, [])] + index_batch
        hf1 = lambda x: (len(x),)+x
        tmp0 = sorted(set([tuple(sorted(set(x))) for x in data_i]), key=hf1)
        all_data[key_str] = [' '.join(str(y) for y in x) for x in tmp0]
        with open(file, 'w', encoding='utf-8') as fid:
            json.dump(all_data, fid, indent=2)
    if key_str is None:
        ret = {k:[[int(y) for y in x.split(' ')] for x in v] for k,v in all_data.items()}
    else:
        ret = [[int(y) for y in x.split(' ')] for x in all_data.get(key_str,[])]
    return ret

tmp0 = [
    np.array([[1.0, 0.0], [0.0, 1.0]]),
    np.array([[0.0, 1.0], [1.0, 0.0]]),
    np.array([[0.0, -1j], [1j, 0.0]]),
    np.array([[1.0, 0.0], [0.0, -1.0]]),
]
_one_pauli_str_to_np = dict(zip('IXYZ', tmp0))

@functools.lru_cache
def get_pauli_group(num_qubit, /, kind='numpy', use_sparse=False):
    assert kind in {'numpy','str','str_to_index'}
    if use_sparse:
        assert kind=='numpy'
    if kind=='numpy':
        if use_sparse:
            # @20230309 scipy.sparse.kron have not yet been ported https://docs.scipy.org/doc/scipy/reference/sparse.html
            hf0 = lambda x,y: scipy.sparse.coo_array(scipy.sparse.kron(x,y,format='coo'))
            hf_kron = lambda x: functools.reduce(hf0, x)
            tmp0 = [scipy.sparse.coo_array(_one_pauli_str_to_np[x]) for x in 'IXYZ']
            tmp1 = [(0,1,2,3)]*num_qubit
            ret = [hf_kron([tmp0[y] for y in x]) for x in itertools.product(*tmp1)]
            # x = ret[0]
            # x.toarray()[x.row, x.col] #x.data
        else:
            hf_kron = lambda x: functools.reduce(np.kron, x)
            tmp0 = [_one_pauli_str_to_np[x] for x in 'IXYZ']
            tmp1 = [(0,1,2,3)]*num_qubit
            ret = np.stack([hf_kron([tmp0[y] for y in x]) for x in itertools.product(*tmp1)])
    else:
        tmp0 = tuple(''.join(x) for x in itertools.product(*['IXYZ']*num_qubit))
        if kind=='str':
            ret = tmp0
        else: #str_to_index
            ret = {y:x for x,y in enumerate(tmp0)}
    return ret


def pauli_str_to_matrix(pauli_str, return_orth=False):
    #'XX YZ IZ'
    pauli_str = sorted(set(pauli_str.split()))
    num_qubit = len(pauli_str[0])
    assert all(len(x)==num_qubit for x in pauli_str)
    matrix_space = np.stack([hf_kron([_one_pauli_str_to_np[y] for y in x]) for x in pauli_str])
    if return_orth:
        pauli_str_orth = sorted(set(get_pauli_group(num_qubit, kind='str')) - set(pauli_str))
        matrix_space_orth = np.stack([hf_kron([_one_pauli_str_to_np[y] for y in x]) for x in pauli_str_orth])
        ret = matrix_space,matrix_space_orth
    else:
        ret = matrix_space
    return ret

def get_matrix_list_indexing(mat_list, index):
    if isinstance(mat_list, np.ndarray):
        index = np.asarray(index)
        assert (mat_list.ndim==3) and (index.ndim==1)
        ret = mat_list[index]
    else:
        ret = [mat_list[x] for x in index]
    return ret


def get_fidelity(rho0, rho1):
    ndim0 = rho0.ndim
    ndim1 = rho1.ndim
    assert (ndim0 in {1,2}) and (ndim1 in {1,2})
    if isinstance(rho0, torch.Tensor):
        if ndim0==1 and ndim1==1:
            ret = torch.abs(torch.vdot(rho0, rho1))**2
        elif ndim0==1 and ndim1==2:
            ret = torch.vdot(rho0, rho1 @ rho0).real
        elif ndim0==2 and ndim1==1:
            ret = torch.vdot(rho1, rho0 @ rho1).real
        else:
            EVL0,EVC0 = torch.linalg.eigh(rho0)
            zero = torch.tensor(0.0, device=rho0.device)
            tmp0 = torch.sqrt(torch.maximum(zero, EVL0))
            tmp1 = (tmp0.reshape(-1,1) * EVC0.T.conj()) @ rho1 @ (EVC0 * tmp0)
            tmp2 = torch.linalg.eigvalsh(tmp1)
            ret = torch.sum(torch.sqrt(torch.maximum(zero, tmp2)))**2
    else:
        if ndim0==1 and ndim1==1:
            ret = abs(np.vdot(rho0, rho1))**2
        elif ndim0==1 and ndim1==2:
            ret = np.vdot(rho0, rho1 @ rho0).real.item()
        elif ndim0==2 and ndim1==1:
            ret = np.vdot(rho1, rho0 @ rho1).real.item()
        else:
            EVL0,EVC0 = np.linalg.eigh(rho0)
            tmp0 = np.sqrt(np.maximum(0, EVL0))
            tmp1 = (tmp0[:,np.newaxis] * EVC0.T.conj()) @ rho1 @ (EVC0 * tmp0)
            tmp2 = np.linalg.eigvalsh(tmp1)
            ret = np.sum(np.sqrt(np.maximum(0, tmp2)))**2
    return ret


def get_numpy_rng(np_rng_or_seed_or_none=None):
    if np_rng_or_seed_or_none is None:
        ret = np.random.default_rng()
    elif isinstance(np_rng_or_seed_or_none, np.random.Generator):
        ret = np_rng_or_seed_or_none
    else:
        seed = int(np_rng_or_seed_or_none)
        ret = np.random.default_rng(seed)
    return ret


def _random_complex(*size, seed=None):
    np_rng = get_numpy_rng(seed)
    ret = np_rng.normal(size=size + (2,)).astype(np.float64, copy=False).view(np.complex128).reshape(size)
    return ret

def rand_haar_state(dim, tag_complex=True, seed=None):
    r'''Return a random state vector from the Haar measure on the unit sphere in $\mathbb{C}^{d}$.

    $$\left\{ |\psi \rangle \in \mathbb{C} ^d\,\,: \left\| |\psi \rangle \right\| _2=1 \right\}$$

    Parameters:
        dim (int): The dimension of the Hilbert space that the state should be sampled from.
        tag_complex (bool): If True, use complex normal distribution. If False, use real normal distribution.
        seed ([None], int, numpy.RandomState): If int or RandomState, use it for RNG. If None, use default RNG.

    Returns:
        ret (numpy.ndarray): shape=(`dim`,), dtype=np.complex128
    '''
    # http://www.qetlab.com/RandomStateVector
    ret = _random_complex(dim, seed=seed)
    if tag_complex:
        ret = _random_complex(dim, seed=seed)
    else:
        np_rng = get_numpy_rng(seed)
        ret = np_rng.normal(size=dim)
    ret /= np.linalg.norm(ret)
    return ret
