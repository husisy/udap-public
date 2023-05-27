import time
import collections
import numpy as np
import torch
import scipy.sparse
import concurrent.futures
import multiprocessing
import cvxpy

from utils import save_index_to_file, get_matrix_list_indexing

# cannot be torch.linalg.norm()**2 nan when calculating the gradient when norm is almost zero
# see https://github.com/pytorch/pytorch/issues/99868
# hf_torch_norm_square = lambda x: torch.dot(x.conj(), x).real
hf_torch_norm_square = lambda x: torch.sum((x.conj() * x).real)

def hf_tuple_of_any(x, type_=None):
    hf0 = lambda x: x if (type_ is None) else type_(x)
    if isinstance(x,collections.abc.Iterable):
        if isinstance(x, np.ndarray):
            ret = [hf0(y) for y in np.nditer(x)]
        else:
            # error when x is np.array(0)
            ret = tuple(hf0(y) for y in x)
    else:
        ret = hf0(x),
    return ret

hf_tuple_of_int = lambda x: hf_tuple_of_any(x, type_=int)

def matrix_to_gellmann_basis(A, norm_I='sqrt(2/d)'):
    shape0 = A.shape
    N0 = shape0[-1]
    assert norm_I in {'1/d','sqrt(2/d)'}
    factor_I = (1/N0) if norm_I=='1/d' else 1/np.sqrt(2*N0)
    assert len(shape0)>=2 and shape0[-2]==N0
    A = A.reshape(-1,N0,N0)
    if isinstance(A, torch.Tensor):
        indU0,indU1 = torch.triu_indices(N0, N0, offset=1)
        aS = (A + A.transpose(1,2))[:,indU0,indU1]/2
        aA = (A - A.transpose(1,2))[:,indU0,indU1] * (0.5j)
        tmp0 = torch.diagonal(A, dim1=1, dim2=2)
        tmp1 = torch.sqrt(2*torch.arange(1,N0,dtype=torch.float64)*torch.arange(2,N0+1))
        aD = (torch.cumsum(tmp0,dim=1)[:,:-1] - torch.arange(1,N0)*tmp0[:,1:])/tmp1
        aI = torch.einsum(A, [0,1,1], [0]) * factor_I
        ret = torch.concat([aS,aA,aD,aI.view(-1,1)], dim=1)
    else:
        indU0,indU1 = np.triu_indices(N0,1)
        aS = (A + A.transpose(0,2,1))[:,indU0,indU1]/2
        aA = (A - A.transpose(0,2,1))[:,indU0,indU1] * (0.5j)
        tmp0 = np.diagonal(A, axis1=1, axis2=2)
        tmp1 = np.sqrt(2*np.arange(1,N0)*np.arange(2,N0+1))
        aD = (np.cumsum(tmp0,axis=1)[:,:-1] - np.arange(1,N0)*tmp0[:,1:])/tmp1
        aI = np.trace(A, axis1=1, axis2=2) * factor_I
        ret = np.concatenate([aS,aA,aD,aI[:,np.newaxis]], axis=1)
    if len(shape0)==2:
        ret = ret[0]
    else:
        ret = ret.reshape(*shape0[:-2], -1)
    return ret


def gellmann_basis_to_matrix(vec, norm_I='sqrt(2/d)'):
    # I changed the default value norm_I='1/d' to norm_I='sqrt(2/d)' someday, this could lead to some bugs
    shape = vec.shape
    vec = vec.reshape(-1, shape[-1])
    N0 = vec.shape[0]
    N1 = int(np.sqrt(vec.shape[1]))
    assert norm_I in {'1/d','sqrt(2/d)'}
    # 'sqrt(2/d)' tr(Mi Mj)= 2 delta_ij
    factor_I = 1 if norm_I=='1/d' else np.sqrt(2/N1)
    vec0 = vec[:,:(N1*(N1-1)//2)]
    vec1 = vec[:,(N1*(N1-1)//2):(N1*(N1-1))]
    vec2 = vec[:,(N1*(N1-1)):-1]
    vec3 = vec[:,-1:] * factor_I
    assert vec.shape[1]==N1*N1
    if isinstance(vec, torch.Tensor):
        indU0,indU1 = torch.triu_indices(N1,N1,1)
        indU01 = torch.arange(N1*N1).reshape(N1,N1)[indU0,indU1]
        ind0 = torch.arange(N1)
        indU012 = (((N1*N1)*torch.arange(N0).view(-1,1)) + indU01).view(-1)
        zero0 = torch.zeros(N0*N1*N1, dtype=torch.complex128)
        ret0 = torch.scatter(zero0, 0, indU012, (vec0 - 1j*vec1).view(-1)).reshape(N0, N1, N1)
        ret1 = torch.scatter(zero0, 0, indU012, (vec0 + 1j*vec1).view(-1)).reshape(N0, N1, N1).transpose(1,2)
        tmp0 = torch.sqrt(torch.tensor(2,dtype=torch.float64)/(ind0[1:]*(ind0[1:]+1)))
        tmp1 = torch.concat([tmp0*vec2, vec3], axis=1)
        ret2 = torch.diag_embed(tmp1 @ ((ind0.view(-1,1)>=ind0) + torch.diag(-ind0[1:], diagonal=1)).to(tmp1.dtype))
        ret = ret0 + ret1 + ret2
    else:
        ret = np.zeros((N0,N1,N1), dtype=np.complex128)
        indU0,indU1 = np.triu_indices(N1,1)
        # indL0,indL1 = np.tril_indices(N1,-1)
        ind0 = np.arange(N1, dtype=np.int64)
        ret[:,indU0,indU1] = vec0 - 1j*vec1
        tmp0 = np.zeros_like(ret)
        tmp0[:,indU0,indU1] = vec0 + 1j*vec1
        ret += tmp0.transpose(0,2,1)
        tmp1 = np.concatenate([np.sqrt(2/(ind0[1:]*(ind0[1:]+1)))*vec2, vec3], axis=1)
        ret[:,ind0,ind0] = tmp1 @ ((ind0[:,np.newaxis]>=ind0) + np.diag(-ind0[1:], k=1))
    ret = ret[0] if (len(shape)==1) else ret.reshape(*shape[:-1], N1, N1)
    return ret


def reduce_vector_space(np0, zero_eps=1e-10):
    # span_R(R^n) span_C(C^n)
    assert np0.ndim==2
    _,S,V = np.linalg.svd(np0, full_matrices=False)
    ret = V[:(S>zero_eps).sum()]
    return ret


def get_vector_orthogonal_basis(np0, tag_reduce=True, zero_eps=1e-10):
    # span_R(R^n) span_C(C^n)
    assert np0.ndim==2
    if tag_reduce:
        np0 = reduce_vector_space(np0, zero_eps)
    else:
        assert np.abs(np0.conj() @ np0.T - np.eye(np0.shape[0])).max() < np.sqrt(zero_eps)
    N0,N1 = np0.shape
    if N0==N1:
        ret = np.zeros((0,N1), dtype=np0.dtype)
    else:
        _,EVC = np.linalg.eigh(np.eye(N1) - np0.T @ np0.conj())
        ret = EVC[:,N0:].T
    return ret



def get_matrix_orthogonal_basis(np0, field, zero_eps=1e-10):
    # R_T C_T R C C_H R_cT R_c
    # span_R(R_T^nn)/R_T
    # span_C(R_T^nn) span_C(C_T) / C_T
    # span_R(R^mn)/R
    # span_C(R^mn) span_C(C^mn) / C
    # span_R(C_H^nn)/C_H
    # span_R(C_T^nn)/R_cT
    # span_R(C^mn)/R_c
    # (ret0)matrix_subspace(np,(N0,N1,N1))
    # (ret1)matrix_subspace_orth(np,(N2,N1,N1))
    # (ret2)space_char(str)
    assert np0.ndim==3
    assert field in {'real','complex'}
    np.iscomplexobj(np0)
    N0,N1,N2 = np0.shape
    is_hermitian = (N1==N2) and (np.abs(np0-np0.transpose(0,2,1).conj()).max() < zero_eps)
    assert is_hermitian
    if (field=='real') and is_hermitian: #span_R(C_H^nn)
        tmp0 = matrix_to_gellmann_basis(np0).real
        tmp1 = reduce_vector_space(tmp0, zero_eps)
        tmp2 = get_vector_orthogonal_basis(tmp1, tag_reduce=False)
        basis = gellmann_basis_to_matrix(tmp1)
        basis_orth = gellmann_basis_to_matrix(tmp2)
        ret = basis,basis_orth,'C_H'
    return ret


def find_closest_vector_in_space(space, vec, field):
    # field==None: span_R(R)=R span_C(C)=C span_C(C)=R span_C(R)=C
    assert space.ndim>=2
    if space.ndim>2:
        space = space.reshape(space.shape[0], -1)
    vec = vec.reshape(-1)
    assert space.shape[1]==vec.shape[0]
    assert field in {'real','complex'}
    tag0 = np.iscomplexobj(space)
    tag1 = np.iscomplexobj(vec)
    key = ('R' if (field=='real') else 'C') + ('C' if tag0 else 'R') + ('C' if tag1 else 'R')
    if key in {'RRR', 'CCC', 'CRR', 'CRC', 'CCR'}:
        coeff,residuals,_,_ = np.linalg.lstsq(space.T, vec, rcond=None)
    elif key in {'RCR', 'RCC', 'RRC'}:
        tmp0 = np.concatenate([space.real, space.imag], axis=1)
        tmp1 = np.concatenate([vec.real, vec.imag], axis=0)
        coeff,residuals,_,_ = np.linalg.lstsq(tmp0.T, tmp1, rcond=None)
    ret = coeff, residuals.item()
    return ret

def real_matrix_to_special_unitary(matA, tag_real=False):
    assert matA.shape[-1]==matA.shape[-2]
    shape = matA.shape
    matA = matA.reshape(-1, shape[-1], shape[-1])
    if isinstance(matA, torch.Tensor):
        if tag_real:
            tmp0 = torch.triu(matA, 1)
            tmp1 = tmp0 - tmp0.transpose(1,2)
            # torch.linalg.matrix_exp for a batch of input will lead to memory issue, so use torch.stack()
            ret = torch.stack([torch.linalg.matrix_exp(tmp1[x]) for x in range(len(tmp1))])
        else:
            tmp0 = torch.tril(matA, -1)
            tmp1 = torch.triu(matA)
            tmp2 = torch.diagonal(tmp1, dim1=-2, dim2=-1).mean(dim=1).reshape(-1,1,1)
            tmp3 = tmp1 - tmp2*torch.eye(shape[-1], device=matA.device)
            tmp4 = 1j*(tmp0 - tmp0.transpose(1,2)) + (tmp3 + tmp3.transpose(1,2))
            ret = torch.stack([torch.linalg.matrix_exp(1j*tmp4[x]) for x in range(len(tmp4))])
    else:
        if tag_real:
            tmp0 = np.triu(matA, 1)
            tmp1 = tmp0 - tmp0.transpose(0,2,1)
            ret = np.stack([scipy.linalg.expm(x) for x in tmp1])
            # ret = scipy.linalg.expm(tmp1) #TODO scipy-v1.9
        else:
            tmp0 = np.tril(matA, -1)
            tmp1 = np.triu(matA)
            tmp1 = tmp1 - (np.trace(tmp1, axis1=-2, axis2=-1).reshape(-1,1,1)/shape[-1])*np.eye(shape[-1])
            tmp2 = 1j*(tmp0 - tmp0.transpose(0,2,1)) + (tmp1 + tmp1.transpose(0,2,1))
            ret = np.stack([scipy.linalg.expm(1j*x) for x in tmp2])
            # ret = scipy.linalg.expm(1j*tmp2) #TODO scipy-v1.9
    ret = ret.reshape(shape)
    return ret


def _get_sorted_parameter(model):
    tmp0 = sorted([(k,v) for k,v in model.named_parameters() if v.requires_grad], key=lambda x:x[0])
    ret = [x[1] for x in tmp0]
    return ret


def get_model_flat_parameter(model):
    tmp0 = _get_sorted_parameter(model)
    ret = np.concatenate([x.detach().cpu().numpy().reshape(-1) for x in tmp0])
    return ret


def get_model_flat_grad(model):
    tmp0 = _get_sorted_parameter(model)
    ret = np.concatenate([x.grad.detach().cpu().numpy().reshape(-1) for x in tmp0])
    return ret


def set_model_flat_parameter(model, theta, index01=None):
    theta = torch.tensor(theta)
    parameter_sorted = _get_sorted_parameter(model)
    if index01 is None:
        tmp0 = np.cumsum(np.array([0] + [x.numel() for x in parameter_sorted])).tolist()
        index01 = list(zip(tmp0[:-1],tmp0[1:]))
    for ind0,(x,y) in enumerate(index01):
        tmp0 = theta[x:y].reshape(*parameter_sorted[ind0].shape)
        if not parameter_sorted[ind0].is_cuda:
            tmp0 = tmp0.cpu()
        parameter_sorted[ind0].data[:] = tmp0


def hf_model_wrapper(model):
    parameter_sorted = _get_sorted_parameter(model)
    tmp0 = np.cumsum(np.array([0] + [x.numel() for x in parameter_sorted])).tolist()
    index01 = list(zip(tmp0[:-1],tmp0[1:]))
    def hf0(theta, tag_grad=True):
        set_model_flat_parameter(model, theta, index01)
        if tag_grad:
            loss = model()
            for x in parameter_sorted:
                if x.grad is not None:
                    x.grad.zero_()
            if hasattr(model, 'grad_backward'): #designed for custom automatic differentiation
                model.grad_backward(loss)
            else:
                loss.backward() #if no .grad_backward() method, it should be a normal torch.nn.Module
            # scipy.optimize.LBFGS does not support float32 @20221118
            grad = np.concatenate([x.grad.detach().cpu().numpy().reshape(-1).astype(theta.dtype) for x in parameter_sorted])
        else:
            # TODO, if tag_grad=False, maybe we should return fval only, not (fval,None)
            with torch.no_grad():
                loss = model()
            grad = None
        return loss.item(), grad
    return hf0


def hf_callback_wrapper(hf_fval, state:dict=None, print_freq:int=1):
    if state is None:
        state = dict()
    state['step'] = 0
    state['time'] = time.time()
    state['fval'] = []
    state['time_history'] = []
    def hf0(theta):
        step = state['step']
        if (print_freq>0) and (step%print_freq==0):
            t0 = state['time']
            t1 = time.time()
            fval = hf_fval(theta, tag_grad=False)[0]
            print(f'[step={step}][time={t1-t0:.3f} seconds] loss={fval}')
            state['fval'].append(fval)
            state['time'] = t1
            state['time_history'].append(t1-t0)
        state['step'] += 1
    return hf0


def _get_hf_theta(np_rng, key=None):
    if key is None:
        key = ('uniform', -1, 1)
    if isinstance(key, str):
        if key=='uniform':
            key = ('uniform', -1, 1)
        elif key=='normal':
            key = ('normal', 0, 1)
    if isinstance(key, np.ndarray):
        hf_theta = lambda *x: key
    elif hasattr(key, '__len__') and (len(key)>0) and isinstance(key[0], str):
        if key[0]=='uniform':
            hf_theta = lambda *x: np_rng.uniform(key[1], key[2], size=x)
        elif key[0]=='normal':
            hf_theta = lambda *x: np_rng.normal(key[1], key[2], size=x)
        else:
            assert False, f'un-recognized key "{key}"'
    elif callable(key):
        hf_theta = lambda size: key(size, np_rng)
    else:
        assert False, f'un-recognized key "{key}"'
    return hf_theta


def minimize(model, theta0=None, num_repeat=3, tol=1e-7, print_freq=-1, method='L-BFGS-B',
            print_every_round=1, maxiter=None, early_stop_threshold=None, return_all_result=False, seed=None):
    np_rng = np.random.default_rng(seed)
    hf_theta = _get_hf_theta(np_rng, theta0)
    num_parameter = len(get_model_flat_parameter(model))
    hf_model = hf_model_wrapper(model)
    theta_optim_list = []
    theta_optim_best = None
    options = dict() if maxiter is None else {'maxiter':maxiter}
    for ind0 in range(num_repeat):
        theta0 = hf_theta(num_parameter)
        hf_callback = hf_callback_wrapper(hf_model, print_freq=print_freq)
        theta_optim = scipy.optimize.minimize(hf_model, theta0, jac=True, method=method, tol=tol, callback=hf_callback, options=options)
        if return_all_result:
            theta_optim_list.append(theta_optim)
        if (theta_optim_best is None) or (theta_optim.fun<theta_optim_best.fun):
            theta_optim_best = theta_optim
        if (print_every_round>0) and (ind0%print_every_round==0):
            print(f'[round={ind0}] min(f)={theta_optim_best.fun}, current(f)={theta_optim.fun}')
        if (early_stop_threshold is not None) and (theta_optim_best.fun<=early_stop_threshold):
            break
    hf_model(theta_optim_best.x, tag_grad=False) #set theta and model.property (sometimes)
    ret = (theta_optim_best,theta_optim_list) if return_all_result else theta_optim_best
    return ret


class DetectRankModel(torch.nn.Module):
    def __init__(self, basis_orth, rank, dtype='float64', device='cpu'):
        super().__init__()
        self.is_torch = isinstance(basis_orth, torch.Tensor)
        self.use_sparse = self.is_torch and basis_orth.is_sparse #use sparse only when is a torch.tensor
        assert basis_orth.ndim==3
        assert dtype in {'float32','float64'}
        self.dtype = torch.float32 if dtype=='float32' else torch.float64
        self.cdtype = torch.complex64 if dtype=='float32' else torch.complex128
        self.device = device
        self.basis_orth_conj = self._setup_basis_orth_conj(basis_orth)
        self.theta = self._setup_parameter(basis_orth.shape[1], rank, self.dtype, self.device)

        self.matH = None

    def _setup_basis_orth_conj(self, basis_orth):
        # <A,B>=tr(AB^H)=sum_ij (A_ij, conj(B_ij))
        dtype = self.cdtype
        if self.use_sparse:
            assert self.is_torch
            assert self.device=='cpu', f'sparse tensor not support device "{self.device}"'
            index = basis_orth.indices()
            shape = basis_orth.shape
            tmp0 = torch.stack([index[0], index[1]*shape[2] + index[2]])
            basis_orth_conj = torch.sparse_coo_tensor(tmp0, basis_orth.values().conj().to(dtype), (shape[0], shape[1]*shape[2]))
        else:
            if self.is_torch:
                basis_orth_conj = basis_orth.conj().reshape(basis_orth.shape[0],-1).to(device=self.device, dtype=dtype)
            else:
                basis_orth_conj = torch.tensor(basis_orth.conj().reshape(basis_orth.shape[0],-1), dtype=dtype, device=self.device)
        return basis_orth_conj

    def _setup_parameter(self, dim0, rank, dtype, device):
        np_rng = np.random.default_rng()
        rank = hf_tuple_of_int(rank)
        hf0 = lambda *x: torch.nn.Parameter(torch.tensor(np_rng.uniform(-1,1,size=x), dtype=dtype, device=device))
        assert len(rank)==3
        assert all(x>=0 for x in rank) and (1<=sum(rank)) and (sum(rank)<=dim0)
        theta = {
            'unitary0':hf0(dim0, dim0),
            'EVL_free':hf0(rank[0]) if (rank[0]>0) else None,
            'EVL_positive':hf0(rank[1]) if (rank[1]>0) else None,
            'EVL_negative':hf0(rank[2]) if (rank[2]>0) else None,
        }
        ret = torch.nn.ParameterDict(theta)
        return ret

    def forward(self):
        theta = self.theta
        tmp0 = [
            theta['EVL_free'],
            None if (theta['EVL_positive'] is None) else torch.nn.functional.softplus(theta['EVL_positive']),
            None if (theta['EVL_negative'] is None) else (-torch.nn.functional.softplus(theta['EVL_negative'])),
        ]
        tmp1 = torch.cat([x for x in tmp0 if x is not None])
        EVL = tmp1 / torch.linalg.norm(tmp1)
        unitary = real_matrix_to_special_unitary(theta['unitary0'], tag_real=False)[:len(EVL)]
        matH = (unitary.T.conj()*EVL) @ unitary
        loss = hf_torch_norm_square(self.basis_orth_conj @ matH.reshape(-1))
        self.matH = matH
        return loss

    def get_matrix(self, theta, matrix_subspace):
        set_model_flat_parameter(self, theta)
        with torch.no_grad():
            self()
        matH = self.matH.detach().cpu().numpy().copy()
        coeff, residual = find_closest_vector_in_space(matrix_subspace, matH, field='real')
        return matH,coeff,residual


class DetectUDPModel(torch.nn.Module):
    def __init__(self, basis_orth, dtype='float32', device='cpu'):
        super().__init__()
        self.is_torch = isinstance(basis_orth, torch.Tensor)
        self.use_sparse = self.is_torch and basis_orth.is_sparse #use sparse only when is a torch.tensor
        assert basis_orth.ndim==3
        assert dtype in {'float32','float64'}
        self.dtype = torch.float32 if dtype=='float32' else torch.float64
        self.cdtype = torch.complex64 if dtype=='float32' else torch.complex128
        self.device = device
        self.basis_orth_conj = self._setup_basis_orth_conj(basis_orth)
        np_rng = np.random.default_rng()
        hf0 = lambda *size: torch.nn.Parameter(torch.tensor(np_rng.uniform(-1, 1, size=size), dtype=self.dtype))
        self.theta = hf0(4, basis_orth[0].shape[0])
        self.EVL = hf0(2)
        self.matH = None

    def _setup_basis_orth_conj(self, basis_orth):
        # <A,B>=tr(AB^H)=sum_ij (A_ij, conj(B_ij))
        if self.use_sparse:
            assert self.is_torch
            assert self.device=='cpu', f'sparse tensor not support device "{self.device}"'
            index = basis_orth.indices()
            shape = basis_orth.shape
            tmp0 = torch.stack([index[0], index[1]*shape[2] + index[2]])
            basis_orth_conj = torch.sparse_coo_tensor(tmp0, basis_orth.values().conj().to(self.cdtype), (shape[0], shape[1]*shape[2]))
        else:
            if self.is_torch:
                basis_orth_conj = basis_orth.conj().reshape(basis_orth.shape[0],-1).to(device=self.device, dtype=self.cdtype)
            else:
                basis_orth_conj = torch.tensor(basis_orth.conj().reshape(basis_orth.shape[0],-1), dtype=self.cdtype, device=self.device)
        return basis_orth_conj

    def forward(self):
        tmp0 = self.theta[0] + 1j*self.theta[1]
        EVC0 = tmp0 / torch.linalg.norm(tmp0)
        tmp0 = self.theta[2] + 1j*self.theta[3]
        tmp0 = tmp0 - torch.dot(EVC0.conj(), tmp0) * EVC0
        EVC1 = tmp0 / torch.linalg.norm(tmp0)
        tmp0 = torch.nn.functional.softplus(self.EVL)
        EVL = tmp0 / torch.linalg.norm(tmp0)
        matH = EVC0.reshape(-1,1)*(EVC0.conj()*EVL[0]) - EVC1.reshape(-1,1)*(EVC1.conj()*EVL[1])
        self.matH = matH
        loss = hf_torch_norm_square(self.basis_orth_conj @ matH.reshape(-1))
        return loss


def _check_UDA_UDP_matrix_subspace_one(is_uda, matB, num_repeat, converge_tol,
            early_stop_threshold, udp_use_vector_model, dtype, tag_single_thread):
    if tag_single_thread and torch.get_num_threads()!=1:
        torch.set_num_threads(1)
    if len(matB)==0:
        ret = True,np.inf
    else:
        rank = (0,matB[0].shape[0]-1,1) if is_uda else (0,1,1)
        if not isinstance(matB, np.ndarray): #sparse matrix
            index = np.concatenate([np.stack([x*np.ones(len(y.row),dtype=np.int64), y.row, y.col]) for x,y in enumerate(matB)], axis=1)
            value = np.concatenate([x.data for x in matB])
            matB = torch.sparse_coo_tensor(index, value, (len(matB), *matB[0].shape)).coalesce()
        if udp_use_vector_model:
            model = DetectUDPModel(matB, dtype)
        else:
            model = DetectRankModel(matB, rank=rank, dtype=dtype)
        theta_optim = minimize(model, theta0='normal', num_repeat=num_repeat,
                tol=converge_tol, early_stop_threshold=early_stop_threshold, print_every_round=0, print_freq=0)
        ret = theta_optim.fun>early_stop_threshold, theta_optim.fun
        # always assume that identity is measured, and matrix subspace A is traceless, so no need to test loss(0,n,0)
    return ret


def _check_UDA_UDP_matrix_subspace_parallel(is_uda, matB, num_repeat, converge_tol,
            early_stop_threshold, udp_use_vector_model, dtype, num_worker, tag_single_thread):
    if isinstance(matB, np.ndarray) or scipy.sparse.issparse(matB[0]):
        is_single_item = True
        if isinstance(matB, np.ndarray):
            assert (matB.ndim==3) and (matB.shape[1]==matB.shape[2])
            matB_list = [matB]
        else:
            assert all((x.shape[0]==x.shape[1]) and (x.format=='coo') for x in matB)
            matB_list = [matB]
    else:
        is_single_item = False
        if isinstance(matB[0], np.ndarray):
            assert all(((x.ndim==3) and (x.shape[1]==x.shape[2])) for x in matB)
        else:
            assert all((y.shape[0]==y.shape[1]) and (y.format=='coo') for x in matB for y in x)
        matB_list = matB
    assert len(matB_list)>0
    kwargs = {'is_uda':is_uda, 'num_repeat':num_repeat, 'converge_tol':converge_tol, 'early_stop_threshold':early_stop_threshold,
            'udp_use_vector_model':udp_use_vector_model, 'dtype':dtype, 'tag_single_thread':tag_single_thread}
    num_worker = min(num_worker, len(matB_list))
    if num_worker == 1:
        time_start = time.time()
        num_pass = 0
        ret = []
        for matB in matB_list:
            ret.append(_check_UDA_UDP_matrix_subspace_one(matB=matB, **kwargs))
            if ret[-1][0]:
                tmp0 = time.time()-time_start
                num_pass = num_pass + 1
                print(f'[{tmp0:.1f}] {num_pass}/{len(ret)}/{len(matB_list)}')
    else:
        # https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker, mp_context=multiprocessing.get_context('spawn')) as executor:
            job_list = [executor.submit(_check_UDA_UDP_matrix_subspace_one, matB=x, **kwargs) for x in matB_list]
            jobid_to_result = dict()
            time_start = time.time()
            num_pass = 0
            for job_i in concurrent.futures.as_completed(job_list):
                ret_i = job_i.result()
                jobid_to_result[id(job_i)] = ret_i
                if ret_i[0]:
                    tmp0 = time.time()-time_start
                    num_pass = num_pass + 1
                    print(f'[{tmp0:.1f}] {num_pass}/{len(jobid_to_result)}/{len(job_list)}')
            ret = [jobid_to_result[id(x)] for x in job_list]
    if is_single_item:
        ret = ret[0]
    return ret

def check_UDA_matrix_subspace(matB, num_repeat, converge_tol=1e-5, early_stop_threshold=1e-2, dtype='float32',
                                    udp_use_vector_model=False, num_worker=1, tag_single_thread=True):
    is_uda = True
    udp_use_vector_model = False #ignore this parameter
    ret = _check_UDA_UDP_matrix_subspace_parallel(is_uda, matB, num_repeat, converge_tol,
            early_stop_threshold, udp_use_vector_model, dtype, num_worker, tag_single_thread)
    return ret


def check_UDP_matrix_subspace(matB, num_repeat, converge_tol=1e-5, early_stop_threshold=1e-2, dtype='float32',
                                    udp_use_vector_model=False, num_worker=1, tag_single_thread=True):
    is_uda = False
    ret = _check_UDA_UDP_matrix_subspace_parallel(is_uda, matB, num_repeat, converge_tol,
            early_stop_threshold, udp_use_vector_model, dtype, num_worker, tag_single_thread)
    return ret


def _find_UDA_UDP_over_matrix_basis_one(is_uda, matrix_basis, num_repeat, num_random_select, indexF, tag_reduce,
            early_stop_threshold, converge_tol, last_converge_tol, last_num_repeat,
            udp_use_vector_model, dtype, tag_single_thread, tag_print):
    if tag_single_thread:
        torch.set_num_threads(1)
    if last_converge_tol is None:
        last_converge_tol = converge_tol/10
    if last_num_repeat is None:
        last_num_repeat = num_repeat*5
    np_rng = np.random.default_rng()
    N0 = len(matrix_basis)
    if not isinstance(matrix_basis, np.ndarray): #list of sparse matrix
        assert not tag_reduce, 'tag_reduce=True is not compatible with sparse matrix'

    time_start = time.time()
    if indexF is not None:
        indexF = set([int(x) for x in indexF])
        assert all(0<=x<N0 for x in indexF)
    else:
        indexF = set()
    indexB = set(list(range(N0)))
    kwargs = {'is_uda':is_uda, 'num_repeat':num_repeat, 'converge_tol':converge_tol, 'early_stop_threshold':early_stop_threshold,
        'udp_use_vector_model':udp_use_vector_model, 'dtype':dtype, 'tag_single_thread':False}
    # tag_single_thread is already set
    index_B_minus_F = np.array(sorted(indexB - set(indexF)), dtype=np.int64)
    assert len(index_B_minus_F)>=num_random_select
    while num_random_select>0:
        selectX = set(np_rng.choice(index_B_minus_F, size=num_random_select, replace=False, shuffle=False).tolist())
        matB = get_matrix_list_indexing(matrix_basis, sorted(indexB-selectX))
        if tag_reduce:
            matB,matB_orth,space_char = get_matrix_orthogonal_basis(matB, field='real', zero_eps=1e-10)
            assert space_char in {'R_T','C_H'}
        if (tag_reduce and len(matB_orth)==0) or (_check_UDA_UDP_matrix_subspace_one(matB=matB, **kwargs)[0]):
            indexB = indexB - selectX
            break
    while True:
        tmp0 = sorted(indexB - indexF)
        if len(tmp0)==0:
            break
        selectX = tmp0[np_rng.integers(len(tmp0))]
        matB = get_matrix_list_indexing(matrix_basis, sorted(indexB-{selectX}))
        if tag_reduce:
            matB,matB_orth,space_char = get_matrix_orthogonal_basis(matB, field='real', zero_eps=1e-10)
            assert space_char in {'R_T','C_H'}
        if tag_reduce and (matB_orth.shape[0]==0):
            ret_hfT = True,np.inf
        else:
            ret_hfT = _check_UDA_UDP_matrix_subspace_one(matB=matB, **kwargs)
        if ret_hfT[0]:
            indexB = indexB - {selectX}
            if tag_print:
                tmp0 = time.time() - time_start
                tmp1 = 'loss(n-1,1)' if is_uda else 'loss(1,1)'
                print(f'[{tmp0:.1f}s/{len(indexB)}/{len(indexF)}] {tmp1}={ret_hfT[1]:.5f}')
        else:
            indexF = indexF | {selectX}
    matB = get_matrix_list_indexing(matrix_basis, sorted(indexB))
    kwargs['converge_tol'] = last_converge_tol
    kwargs['num_repeat'] = last_num_repeat
    ret_hfT = _check_UDA_UDP_matrix_subspace_one(matB=matB, **kwargs)
    if tag_print and ret_hfT[0]:
        tmp0 = time.time() - time_start
        tmp1 = 'loss(n-1,1)' if is_uda else 'loss(1,1)'
        print(f'[{tmp0:.1f}s/{len(indexB)}/{len(indexF)}] {tmp1}={ret_hfT[1]:.5f} [{len(indexB)}] {sorted(indexB)}')
    ret = sorted(indexB) if ret_hfT[0] else None
    return ret


def _find_UDA_UDP_over_matrix_basis(is_uda, num_round, matrix_basis, num_repeat, num_random_select, indexF, tag_reduce,
            early_stop_threshold, converge_tol, last_converge_tol, last_num_repeat, udp_use_vector_model,
            dtype, num_worker, key, file, tag_single_thread):
    num_worker = min(num_worker, num_round)
    assert num_worker>=1
    if isinstance(matrix_basis,np.ndarray):
        assert (matrix_basis.ndim==3) and (matrix_basis.shape[1]==matrix_basis.shape[2])
        assert np.abs(matrix_basis-matrix_basis.transpose(0,2,1).conj()).max() < 1e-10
    else:
        # should be scipy.sparse.coo_matrix
        assert not tag_reduce, 'tag_reduce not support sparse data'
        assert all(scipy.sparse.issparse(x) and (x.format=='coo') and (x.shape[0]==x.shape[1]) for x in matrix_basis)
        for x in matrix_basis:
            tmp0 = (x-x.T.conj()).data
            assert (len(tmp0)==0) or np.abs(tmp0).max() < 1e-10
    ret = []
    kwargs = {'is_uda':is_uda, 'matrix_basis':matrix_basis, 'num_repeat':num_repeat, 'num_random_select':num_random_select,
            'indexF':indexF, 'tag_reduce':tag_reduce, 'early_stop_threshold':early_stop_threshold, 'converge_tol':converge_tol,
            'last_converge_tol':last_converge_tol, 'last_num_repeat':last_num_repeat, 'udp_use_vector_model':udp_use_vector_model,
            'dtype':dtype, 'tag_single_thread':tag_single_thread}
    if num_worker==1:
        kwargs['tag_print'] = True
        for _ in range(num_round):
            ret_i = _find_UDA_UDP_over_matrix_basis_one(**kwargs)
            if ret_i is not None:
                ret.append(ret_i)
                if key is not None:
                    assert file is not None
                    save_index_to_file(file, key, ret_i)
    else:
        kwargs['tag_print'] = False
        kwargs['tag_single_thread'] = True
        # https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker, mp_context=multiprocessing.get_context('spawn')) as executor:
            job_list = [executor.submit(_find_UDA_UDP_over_matrix_basis_one, **kwargs) for _ in range(num_round)]
            time_start = time.time()
            for ind0,job_i in enumerate(concurrent.futures.as_completed(job_list)):
                ret_i = job_i.result()
                if ret_i is not None:
                    ret.append(ret_i)
                    tmp0 = time.time() - time_start
                    print(f'[round-{ind0}][{tmp0:.1f}s/{len(ret_i)}] {sorted(ret_i)}')
                    if key is not None:
                        assert file is not None
                        save_index_to_file(file, key, ret_i)
    ret = sorted(ret, key=len)
    return ret


# TODO remove indexF
def find_UDA_over_matrix_basis(num_round, matrix_basis, num_repeat, num_random_select, indexF=None, tag_reduce=True,
            early_stop_threshold=0.01, converge_tol=1e-5, last_converge_tol=None, last_num_repeat=None,
            udp_use_vector_model=False, dtype='float32', num_worker=1, key=None, file=None, tag_single_thread=True):
    is_uda = True
    udp_use_vector_model = False
    ret = _find_UDA_UDP_over_matrix_basis(is_uda, num_round, matrix_basis, num_repeat, num_random_select, indexF, tag_reduce,
            early_stop_threshold, converge_tol, last_converge_tol, last_num_repeat, udp_use_vector_model,
            dtype, num_worker, key, file, tag_single_thread)
    return ret


def find_UDP_over_matrix_basis(num_round, matrix_basis, num_repeat, num_random_select, indexF=None, tag_reduce=True,
            early_stop_threshold=0.01, converge_tol=1e-5, last_converge_tol=None, last_num_repeat=None,
            udp_use_vector_model=False, dtype='float32', num_worker=1, key=None, file=None, tag_single_thread=True):
    is_uda = False
    ret = _find_UDA_UDP_over_matrix_basis(is_uda, num_round, matrix_basis, num_repeat, num_random_select, indexF, tag_reduce,
            early_stop_threshold, converge_tol, last_converge_tol, last_num_repeat, udp_use_vector_model,
            dtype, num_worker, key, file, tag_single_thread)
    return ret


def get_UDA_theta_optim_special_EVC(matB, num_repeat=100, tol=1e-12, early_stop_threshold=1e-10, tag_single_thread=True, print_every_round=0):
    if tag_single_thread and torch.get_num_threads()!=1:
        torch.set_num_threads(1)
    if not isinstance(matB, np.ndarray): #sparse matrix
        index = np.concatenate([np.stack([x*np.ones(len(y.row),dtype=np.int64), y.row, y.col]) for x,y in enumerate(matB)], axis=1)
        value = np.concatenate([x.data for x in matB])
        matB = torch.sparse_coo_tensor(index, value, (len(matB), *matB[0].shape)).coalesce()
    model = DetectRankModel(matB, rank=(0, matB[0].shape[0]-1,1), dtype='float64')
    theta_optim = minimize(model, theta0='normal', num_repeat=num_repeat,
            tol=tol, early_stop_threshold=early_stop_threshold, print_every_round=print_every_round, print_freq=0)
    model()
    matH = model.matH.detach().cpu().numpy().copy()
    EVL,EVC = np.linalg.eigh(matH)
    assert (EVL[0]<=0) and (np.abs(matH @ EVC[:,0] - EVC[:,0]*EVL[0]).max() < 1e-8)
    return theta_optim, EVC[:,0]


def density_matrix_recovery_SDP(op_list, measure, converge_eps=None):
    dim = op_list.shape[1]
    rho = cvxpy.Variable((dim,dim), hermitian=True)
    tmp0 = np.asarray(op_list).reshape(-1, dim*dim).T
    tmp1 = cvxpy.real(cvxpy.reshape(rho, (dim*dim,), order='F') @ tmp0)
    # objective = cvxpy.Minimize(cvxpy.sum_squares(tmp1 - measure))
    objective = cvxpy.Minimize(cvxpy.norm(tmp1-measure, 2))
    constraints = [rho>>0, cvxpy.trace(rho)==1]
    prob = cvxpy.Problem(objective, constraints)
    if converge_eps is not None:
        # TODO mosek is faster
        prob.solve(solver=cvxpy.SCS, eps=converge_eps)
    else:
        prob.solve()
    return np.ascontiguousarray(rho.value), prob.value


hf_chebval_n = lambda x, n: np.polynomial.chebyshev.chebval(x, np.array([0]*n+[1]))*(1 if n==0 else np.sqrt(2))

def get_chebshev_orthonormal(dim_qudit, alpha, with_computational_basis=False, return_basis=False):
    # with_computational_basis=False: 4PB
    # with_computational_basis=True: 5PB
    rootd = np.cos(np.pi*(np.arange(dim_qudit)+0.5)/dim_qudit)
    basis0 = np.stack([hf_chebval_n(rootd, x) for x in range(dim_qudit)], axis=1)/np.sqrt(dim_qudit)

    rootd1 = np.cos(np.pi*(np.arange(dim_qudit-1)+0.5)/(dim_qudit-1))
    tmp1 = np.stack([hf_chebval_n(rootd1, x) for x in range(dim_qudit)], axis=1)/np.sqrt(dim_qudit-1)
    tmp2 = np.array([0]*(dim_qudit-1)+[1])
    basis1 = np.concatenate([tmp1,tmp2[np.newaxis]], axis=0)

    basis2 = np.stack([hf_chebval_n(rootd, x)*np.exp(1j*alpha*x) for x in range(dim_qudit)], axis=1)/np.sqrt(dim_qudit)

    tmp1 = np.stack([hf_chebval_n(rootd1, x)*np.exp(1j*alpha*x) for x in range(dim_qudit)], axis=1)/np.sqrt(dim_qudit-1)
    tmp2 = np.array([0]*(dim_qudit-1)+[1])
    basis3 = np.concatenate([tmp1,tmp2[np.newaxis]], axis=0)

    basis_list = [basis0,basis1,basis2,basis3]
    if with_computational_basis:
        basis_list.append(np.eye(dim_qudit))
        tmp0 = np.eye(dim_qudit)

    tmp0 = np.concatenate(basis_list, axis=0)
    ret = tmp0[:,:,np.newaxis]*(tmp0[:,np.newaxis].conj())
    if return_basis:
        ret = ret,basis_list
    return ret
