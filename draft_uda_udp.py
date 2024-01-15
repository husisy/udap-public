import os
import time
import collections
import itertools
import pickle
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import save_index_to_file, get_pauli_group, get_matrix_list_indexing, pauli_str_to_matrix, get_fidelity, rand_haar_state
from ud_utils import (check_UDA_matrix_subspace, check_UDP_matrix_subspace, find_UDP_over_matrix_basis, get_UDA_theta_optim_special_EVC,
                    density_matrix_recovery_SDP, get_chebshev_orthonormal)

np_rng = np.random.default_rng()
cp_tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']

def pauli_full_json_to_core_json():
    full_json = 'data/pauli-indexB-full.json'
    core_json = 'data/pauli-indexB-core.json'
    z0 = save_index_to_file(full_json)
    pauli_len_dict = {'2':14, '3':35, '4':112, '5':403}
    z1 = dict()
    for key,value in z0.items():
        tmp0 = [x for x in value if len(x)<pauli_len_dict[key]]
        if len(tmp0):
            z1[key] = tmp0
    if os.path.exists(core_json):
        os.rename(core_json, core_json+'.bak')
    for k,v in z1.items():
        _ = save_index_to_file(core_json, k, v)


def demo_print_saved_result():
    z0 = save_index_to_file('data/pauli-indexB-full.json') #not enter git version control, maybe google drive
    z0 = save_index_to_file('data/pauli-indexB-core.json')
    for key in sorted(z0.keys()):
        tmp0 = collections.Counter(len(x) for x in z0[key])
        tmp1 = [f'{tmp0[x]}x{x}' for x in sorted(tmp0.keys())]
        print(f'[{key}]:', ' '.join(tmp1))
    # [2]: 6x11 19x13
    # [3]: 176x31 258x32 316x34 343x35 426x36 329x37 352x38 115x39 14x40 1x41 3x42 1x44
    # [4]: 3x106 14x107 53x108 124x109 305x110 736x111 1678x112 3394x113 6569x114 11629x115 18103x116 26882x117 35611x118 43509x119 48855x120 50711x121 48069x122
    # [5]: 1x393 1x395 2x397 3x398 1x399 3x400 5x401 6x402 13x403 18x404 25x405 27x406 40x407 47x408 52x409 59x410 67x411 80x412 103x413

    # merge v100-pauli.json with pauli-indexB-full.json
    # z1 = save_index_to_file('v100-pauli.json')
    # _ = save_index_to_file('data/pauli-indexB-full.json', '4', z1['4,pauli,udp'])


def demo_pauli_loss_function():
    num_worker = 15
    num_qubit_to_num_repeat = {2:50, 3:50, 4:400, 5:3200}
    add_udp = True
    pauli_len_dict = {2:11, 3:31, 4:106, 5:398}
    z0 = save_index_to_file('data/pauli-indexB-core.json')
    z1 = {int(x0):{y0:list(y1) for y0,y1 in itertools.groupby(x1, key=len)} for x0,x1 in z0.items()}
    # {x0:' '.join([f'{len(y1)}x{y0}' for y0,y1 in x1.items()]) for x0,x1 in z1.items()}
    z2 = dict()
    for num_qubit,max_len in pauli_len_dict.items():
        if num_qubit not in z1:
            continue
        num_repeat = num_qubit_to_num_repeat[num_qubit]
        matrix_subspace = get_pauli_group(num_qubit, use_sparse=True)

        tmp0 = [y for x,y in z1[num_qubit].items() if x<=max_len]
        index_list = [y for x in tmp0 for y in x]
        matB_list = [get_matrix_list_indexing(matrix_subspace, x) for x in index_list]
        uda_loss = [x[1] for x in check_UDA_matrix_subspace(matB_list, num_repeat=num_repeat, num_worker=num_worker)]
        # uda_loss = [0] *len(matB_list)
        if add_udp:
            udp_loss = [x[1] for x in check_UDP_matrix_subspace(matB_list, num_repeat=num_repeat, num_worker=num_worker)]
        else:
            udp_loss = [0]*len(uda_loss)
        tmp0 = itertools.groupby(zip(index_list,uda_loss,udp_loss), key=lambda x:len(x[0]))
        tmp1 = {x0:list(zip(*x1)) for x0,x1 in tmp0}
        for x0,x1 in tmp1.items():
            print(f'[{num_qubit},{x0},UDA]', np.sort(np.array(x1[1])))
            if add_udp:
                print(f'[{num_qubit},{x0},UDP]', np.sort(np.array(x1[2])))
        z2[num_qubit] = tmp1


def demo_search_UD_in_pauli_group():
    num_qubit = 3
    num_repeat = {2:10, 3:10, 4:80, 5:640}[num_qubit]
    num_random_select = {2:0, 3:10, 4:80, 5:400}[num_qubit]
    matrix_subspace = get_pauli_group(num_qubit, use_sparse=True)
    kwargs = {'num_repeat':num_repeat,  'num_random_select':num_random_select, 'indexF':[0],
                'num_worker':30, 'udp_use_vector_model':True, 'tag_reduce':False, 'key':str(num_qubit),
                'file':'tbd00.json'}
    z0 = find_UDP_over_matrix_basis(num_round=1, matrix_basis=matrix_subspace, **kwargs)
    # key '{num_qubit}': find via UDP, but check use UDA
    # matB_list = [get_matrix_list_indexing(matrix_subspace, x) for x in z0]
    # z1 = check_UDA_matrix_subspace(matB_list, num_repeat=num_repeat*5, num_worker=19)


def demo_search_UD_in_pauli_group_timing():
    num_round = 10
    info_list = []
    for num_qubit in [2,3,4,5]:
        num_repeat = {2:10, 3:10, 4:80, 5:640}[num_qubit]
        num_random_select = {2:0, 3:10, 4:80, 5:400}[num_qubit]
        kwargs = {'num_repeat':num_repeat,  'num_random_select':num_random_select, 'indexF':[0],
                    'num_worker':1, 'udp_use_vector_model':True, 'tag_reduce':False, 'key':str(num_qubit),
                    'file':'tbd00.json'}
        matrix_subspace = get_pauli_group(num_qubit, use_sparse=True)
        t0 = time.time()
        z0 = find_UDP_over_matrix_basis(num_round=num_round, matrix_basis=matrix_subspace, **kwargs)
        tmp0 = time.time()-t0
        info_list.append((num_qubit, tmp0, len(z0)))
    for num_qubit,tmp0,tmp1 in info_list:
        # only count the time of successful cases
        print(f'[#qubit={num_qubit}] TotalTime={tmp0:.3f}s, AverageTime={tmp0/tmp1:.3f}s')
        # [#qubit=2] TotalTime=3.993s, AverageTime=0.399s
        # [#qubit=3] TotalTime=24.778s, AverageTime=2.478s
        # [#qubit=4] TotalTime=805.965s, AverageTime=80.597s
        # [#qubit=5] TotalTime=50382.383s, AverageTime=8397.064s

    info_list = []
    for num_qubit in [2,3,4,5]:
        indexB = save_index_to_file('data/pauli-indexB-core.json', str(num_qubit))[0]
        matrix_subspace = get_pauli_group(num_qubit, use_sparse=True)
        matB = get_matrix_list_indexing(matrix_subspace, indexB)
        num_repeat = {2:10, 3:10, 4:80, 5:640}[num_qubit]
        kwargs = {'num_repeat':num_repeat, 'converge_tol':1e-5, 'early_stop_threshold':1e-2,
                'dtype':'float64', 'num_worker':1, 'udp_use_vector_model':True}
        t0 = time.time()
        tmp0 = check_UDP_matrix_subspace(matB, **kwargs)
        info_list.append((num_qubit, time.time()-t0))
        assert tmp0 #should be UDP/UDA
    for num_qubit,tmp0 in info_list:
        print(f'[#qubit={num_qubit}] Time={tmp0:.3f}s')
    # [#qubit=2] Time=0.048s
    # [#qubit=3] Time=0.057s
    # [#qubit=4] Time=2.406s
    # [#qubit=5] Time=25.175s

    # #qubit, time (s)
    # #qubit=2, 0.05/0.4
    # #qubit=3, 0.06/2.5
    # #qubit=4, 2.4/81
    # #qubit=3, 25/8400

    # CAPTION: in the time column, $t1/t2$ denotes the typical time required
    # to perform one round UDP certification and searching minimum set respectively.



def demo_pauli_2qubit_loss_function():
    matB = pauli_str_to_matrix('II IX IY IZ XI YX YY YZ ZX ZY ZZ')
    assert matB.shape==(11,4,4)
    matB_delete_one_list = [matB[sorted(set(range(11))-{x})] for x in range(11)]
    matB_list = [matB] + matB_delete_one_list
    kwargs = {'num_repeat':250, 'converge_tol':1e-12, 'early_stop_threshold':1e-10, 'dtype':'float64', 'num_worker':12}
    z0 = check_UDA_matrix_subspace(matB_list, **kwargs)
    # [(True, 1.0000000000336475),
    #  (False, 1.2951900553955004e-11),
    #  (False, 4.918275861206689e-11),
    #  (False, 6.7655334551738e-11),
    #  (False, 8.556658846697481e-11),
    #  (False, 8.469010949285224e-11),
    #  (False, 2.933035981318818e-11),
    #  (False, 7.087059997072998e-11),
    #  (False, 4.7286659824272235e-11),
    #  (False, 6.013314584973174e-11),
    #  (False, 6.972959266728439e-11),
    #  (False, 4.3429860254647276e-11)]


def demo_ud_over_pauli_probability():
    # time required (mac-studio,1cpu,1sample,float32): 6s(n=3) 9s(n=4) 20s(n=5) 120s(n=6) 1000s(n=7)
    datapath = 'data/n_pauli_sucess_probability.pkl'
    if os.path.exists(datapath):
        with open(datapath, 'rb') as fid:
            tmp0 = pickle.load(fid)
            exp_parameter_list = tmp0['exp_parameter_list']
            hyper_parameter = tmp0['hyper_parameter']
            num_sample = tmp0['num_sample']
            all_data = tmp0['all_data']
    else:
        tmp0 = np.linspace(0,1,32)[1:-1]
        exp_parameter_list = sorted(set([(x,int(y)) for x in [3,4,5,6,7] for y in (4**x)*tmp0]))
        num_sample = 19*10
        np_rng = np.random.default_rng()
        hyper_parameter = {'num_repeat':80, 'converge_tol':1e-7, 'early_stop_threshold':1e-5,
                    'dtype':'float64', 'num_worker':19, 'udp_use_vector_model':True}
        all_data = []
        time_start = time.time()
        for num_qubit,num_op in exp_parameter_list:
            tmp0 = time.time() - time_start
            print(f'[{tmp0:.1f}s] {num_qubit=} {num_op=}')
            matrix_subspace = get_pauli_group(num_qubit, use_sparse=True)
            matB_list = []
            for _ in range(num_sample):
                index = np.zeros(num_op, dtype=np.int64) #first element must be identity
                index[1:] = np.sort(np_rng.permutation(len(matrix_subspace)-1)[:(num_op-1)])+1
                matB_list.append(get_matrix_list_indexing(matrix_subspace, index))
            tmp0 = check_UDP_matrix_subspace(matB_list, **hyper_parameter)
            all_data.append([x[1] for x in tmp0])
        all_data = np.array(all_data)
        tmp0 = {'exp_parameter_list':exp_parameter_list, 'hyper_parameter':hyper_parameter, 'num_sample':num_sample, 'all_data':all_data}
        with open(datapath, 'wb') as fid:
            pickle.dump(tmp0, fid)

    key_list = sorted({x[0] for x in exp_parameter_list})
    key_list.pop(-1) #bad data
    all_data_dict = dict()
    for key_i in key_list:
        ind0 = [x for x,y in enumerate(exp_parameter_list) if (y[0]==key_i)]
        tmp0 = np.array([exp_parameter_list[x][1] for x in ind0])
        all_data_dict[key_i] = tmp0, all_data[ind0]

    threshold = 0.01
    fig,ax = plt.subplots(figsize=(5,4))
    for key_i in key_list:
        tmp0,tmp1 = all_data_dict[key_i]
        ax.plot(tmp0*100/(4**key_i), (tmp1>threshold).mean(axis=1), marker='.', label=f'n={key_i}')
    ax.set_xlabel('#pauli-op (percent)', fontsize=14)
    ax.set_ylabel('probability', fontsize=14)
    # ax.set_ylabel(r'$\mathrm{mean}(\mathcal{L}_{1,1})$')
    # ax.set_title(f'#sample={num_sample} threshold={threshold}')
    ax.grid()
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.1, 1.1)
    fig.tight_layout()
    ax.legend(fontsize=14)
    # fig.savefig('tbd00.png', dpi=200)
    fig.savefig('data/n_pauli_sucess_probability.png', dpi=200)
    fig.savefig('data/n_pauli_sucess_probability.pdf')


def demo_pauli_stability():
    num_qubit_list = [2,3,4,5]
    noise_rate = 1e-4
    num_random = 500
    cvxpy_eps = 1e-6

    num_qubit_to_data_dict = dict()
    for num_qubit in num_qubit_list:
        print(num_qubit)
        index = save_index_to_file('data/pauli-indexB-core.json', str(num_qubit))[0]
        pauli_group = get_pauli_group(num_qubit, use_sparse=True)
        matrix_subspace = get_matrix_list_indexing(pauli_group, index)
        theta_optim,state_special = get_UDA_theta_optim_special_EVC(matrix_subspace, num_repeat=100, print_every_round=1)
        state_list = [state_special] + [rand_haar_state(2**num_qubit) for _ in range(2)]
        num_qubit_to_data_dict[(num_qubit, 'theta_optim')] = theta_optim
        num_qubit_to_data_dict[(num_qubit, 'state_list')] = state_list
        num_qubit_to_data_dict[(num_qubit, 'index')] = index

    for num_qubit in num_qubit_list:
        index = num_qubit_to_data_dict[(num_qubit, 'index')]
        pauli_group = get_pauli_group(num_qubit, use_sparse=True)
        matrix_subspace = np.stack([x.toarray() for x in get_matrix_list_indexing(pauli_group, index)])
        state_list = num_qubit_to_data_dict[(num_qubit, 'state_list')]
        data = []
        for state_i in state_list:
            measure_no_noise = ((matrix_subspace @ state_i) @ state_i.conj()).real
            for _ in tqdm(range(num_random)):
                tmp0 = np_rng.normal(size=len(matrix_subspace))
                noise = tmp0 * (noise_rate/np.linalg.norm(tmp0))
                tmp1,eps = density_matrix_recovery_SDP(matrix_subspace, measure_no_noise + noise, converge_eps=cvxpy_eps)
                tmp2 = np.linalg.norm(tmp1 - state_i[:,np.newaxis]*state_i.conj(), ord='fro') #frob norm
                tmp3 = get_fidelity(tmp1, state_i)
                data.append((eps, tmp2, tmp3))
        num_qubit_to_data_dict[(num_qubit, 'data')] = np.array(data).reshape(len(state_list), num_random, 3).transpose(0,2,1)
    # with open(f'data/pauli_worst_case_dim.pkl', 'wb') as fid:
    #     tmp0 = dict(num_qubit_to_data_dict=num_qubit_to_data_dict, num_qubit_list=num_qubit_list, noise_rate=noise_rate,
    #                 num_random=num_random, cvxpy_eps=cvxpy_eps)
    #     pickle.dump(tmp0, fid)

    # ydata_list = [num_qubit_to_data_dict[(x,'data')] for x in dim_qudit_list]
    # fig,ax = plt.subplots()
    # tmp0 = [1/np.sqrt(num_qubit_to_data_dict[(x,'theta_optim')].fun) for x in dim_qudit_list]
    # ax.plot(dim_qudit_list, tmp0, 'o-', color=cp_tableau[5], label='1/c')
    # for ind0 in range(ydata_list[0].shape[0]):
    #     tmp0 = np.stack([x[ind0][1]/(x[ind0][0]+noise_rate) for x in ydata_list])
    #     ax.fill_between(dim_qudit_list, tmp0.min(axis=1), tmp0.max(axis=1), alpha=0.2, color=cp_tableau[ind0])
    #     tmp1 = r'$\psi_-$' if ind0==0 else r'random $\sigma_'+f'{ind0-1}$'
    #     ax.plot(dim_qudit_list, tmp0.mean(axis=1), 'o-', color=cp_tableau[ind0], label=tmp1)
    # ax.set_xticks(dim_qudit_list)
    # ax.set_yscale('log')
    # ax.legend()
    # ax.set_xlabel(r'$d$')
    # ax.set_ylabel(r'$\frac{||Y-\sigma||_F}{\epsilon+||f||_2}$')
    # # ax.set_title(r'5PB worst case, $||f||_2=' + f'{noise_rate:.2g}$')
    # fig.tight_layout()
    # fig.savefig('tbd00.png', dpi=200)
    # # fig.savefig('data/pauli_worst_case_dim.png', dpi=200)


def demo_pauli_stability_noise_rate():
    num_qubit = 4
    num_random = 500
    cvxpy_eps = 1e-6
    noise_rate_list = np.logspace(-6, -3, 6)

    with open('data/pauli_worst_case_dim.pkl', 'rb') as fid:
        num_qubit_to_data_dict = pickle.load(fid)['num_qubit_to_data_dict']
        index = num_qubit_to_data_dict[(num_qubit, 'index')]
        pauli_group = get_pauli_group(num_qubit, use_sparse=True)
        matrix_subspace = np.stack([x.toarray() for x in get_matrix_list_indexing(pauli_group, index)])
        state_list = num_qubit_to_data_dict[(num_qubit, 'state_list')]
        theta_optim = num_qubit_to_data_dict[(num_qubit, 'theta_optim')]

    data = []
    for state_i in state_list:
        measure_no_noise = ((matrix_subspace @ state_i) @ state_i.conj()).real
        for noise_rate in noise_rate_list:
            for _ in tqdm(range(num_random)):
                tmp0 = np_rng.normal(size=len(matrix_subspace))
                noise = tmp0 * (noise_rate/np.linalg.norm(tmp0))
                tmp1,eps = density_matrix_recovery_SDP(matrix_subspace, measure_no_noise + noise, converge_eps=cvxpy_eps)
                tmp2 = np.linalg.norm(tmp1 - state_i[:,np.newaxis]*state_i.conj(), ord='fro') #frob norm
                data.append((np.linalg.norm(noise), eps, tmp2))
    data = np.array(data).reshape(-1, len(noise_rate_list), num_random, 3).transpose(0,3,1,2)
    # with open(f'data/pauli_qubit{num_qubit}_noise_rate.pkl', 'wb') as fid:
    #     tmp0 = dict(data=data, num_random=num_random, num_qubit=num_qubit, noise_rate_list=noise_rate_list,
    #                 cvxpy_eps=cvxpy_eps, state_list=state_list, theta_optim=theta_optim)
    #     pickle.dump(tmp0, fid)

    fig,(ax0,ax1) = plt.subplots(1, 2, figsize=(8,4))
    for ind0 in range(data.shape[0]):
        tmp0= noise_rate_list
        tmp1 = data[ind0,1]
        ax0.fill_between(tmp0, tmp1.min(axis=1), tmp1.max(axis=1), alpha=0.2, color=cp_tableau[ind0])
        tmp2 = r'$\psi_-$' if ind0==0 else r'random $\sigma_'+f'{ind0}$'
        ax0.plot(tmp0, tmp1.mean(axis=1), color=cp_tableau[ind0], label=tmp2)

        tmp1 = data[ind0,2] / (data[ind0,0] + data[ind0,1])
        ax1.fill_between(tmp0, tmp1.min(axis=1), tmp1.max(axis=1), alpha=0.2, color=cp_tableau[ind0])
        ax1.plot(tmp0, tmp1.mean(axis=1), color=cp_tableau[ind0])
    ax0.set_ylabel(r'$\epsilon$')
    ax1.set_ylabel(r'$\frac{||Y-\sigma||_F}{\epsilon+||f||_2}$')
    fig.suptitle(f'#qubits={num_qubit}, 1/c={1/np.sqrt(theta_optim.fun):.1f}')
    ax0.legend()
    for ax in [ax0,ax1]:
        ax.set_xlabel(r'$||f||_2$')
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')
    fig.tight_layout()
    fig.savefig('tbd01.png', dpi=200)


def demo_4PB_5PB():
    dim_list = list(range(3, 10))
    hyperparameter = dict(num_repeat=320, early_stop_threshold=1e-10, converge_tol=1e-12, dtype='float64', udp_use_vector_model=False)

    udp_3pb_loss_list = []
    uda_4pb_loss_list = []
    uda_5pb_loss_list = []
    udp_4pb_loss_list = []
    udp_5pb_loss_list = []
    for dim_i in dim_list:
        print(dim_i)
        alpha = np.pi/dim_i

        # 3PB
        matB = get_chebshev_orthonormal(dim_i, alpha, with_computational_basis=False)[:(-dim_i)]
        udp_3pb_loss_list.append(check_UDP_matrix_subspace(matB, **hyperparameter)[1])

        # 4PB
        matB = get_chebshev_orthonormal(dim_i, alpha, with_computational_basis=False)
        # matB = get_matrix_orthogonal_basis(matB, field='real')[0]
        udp_4pb_loss_list.append(check_UDP_matrix_subspace(matB, **hyperparameter)[1])
        uda_4pb_loss_list.append(check_UDA_matrix_subspace(matB, **hyperparameter)[1])

        # 5PB
        matB = get_chebshev_orthonormal(dim_i, alpha, with_computational_basis=True)
        # matB = get_matrix_orthogonal_basis(matB, field='real')[0]
        udp_5pb_loss_list.append(check_UDP_matrix_subspace(matB, **hyperparameter)[1])
        uda_5pb_loss_list.append(check_UDA_matrix_subspace(matB, **hyperparameter)[1])
    # tmp0 = {'hyperparameter':hyperparameter, 'dim_list':dim_list, 'uda_4pb_loss_list':uda_4pb_loss_list,
    #         'uda_5pb_loss_list':uda_5pb_loss_list, 'udp_4pb_loss_list':udp_4pb_loss_list, 'udp_5pb_loss_list':udp_5pb_loss_list,
    #         'udp_3pb_loss_list':udp_3pb_loss_list}
    # with open('data/20230322_4PB_5PB_UDAP.pkl', 'wb') as fid:
    #     pickle.dump(tmp0, fid)

    fig,ax = plt.subplots()
    ax.plot(dim_list, uda_5pb_loss_list, color=cp_tableau[0], linestyle='dashdot', label='5PB UDA')
    ax.plot(dim_list, udp_5pb_loss_list, color=cp_tableau[0], linestyle='solid', label='5PB UDP')
    ax.plot(dim_list, uda_4pb_loss_list, color=cp_tableau[1], linestyle='dashdot', label='4PB UDA')
    ax.plot(dim_list, udp_4pb_loss_list, color=cp_tableau[1], linestyle='solid', label='4PB UDP')
    ax.set_ylabel(r'$\mathcal{L}_{n-1,1}$ or $\mathcal{L}_{1,1}$')
    ax.set_xlabel('qudit $d$')
    ax.set_xticks(np.arange(3,10))
    ax.set_yscale('log')
    ax.legend()
    ax.axhline(1e-9, linestyle='dashed', color=cp_tableau[2])
    ax.set_ylim(5e-12, 5)
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)
    # fig.savefig('data/20230322_4PB_5PB_UDAP.png', dpi=200)


def demo_UDA_noise_rate():
    dim_qudit = 7
    num_random = 500
    cvxpy_eps = 1e-6
    noise_rate_list = np.logspace(-6, -3, 10)

    matrix_subspace = get_chebshev_orthonormal(dim_qudit, alpha=np.pi/dim_qudit, with_computational_basis=True)
    theta_optim,state_special = get_UDA_theta_optim_special_EVC(matrix_subspace, num_repeat=100, tol=1e-13, early_stop_threshold=1e-12)
    state_list = [state_special] + [rand_haar_state(dim_qudit) for _ in range(2)]

    data = []
    for state_i in state_list:
        measure_no_noise = ((matrix_subspace @ state_i) @ state_i.conj()).real
        for noise_rate in noise_rate_list:
            for _ in tqdm(range(num_random)):
                tmp0 = np_rng.normal(size=len(matrix_subspace))
                noise = tmp0 * (noise_rate/np.linalg.norm(tmp0))
                tmp1,eps = density_matrix_recovery_SDP(matrix_subspace, measure_no_noise + noise, converge_eps=cvxpy_eps)
                tmp2 = np.linalg.norm(tmp1 - state_i[:,np.newaxis]*state_i.conj(), ord='fro') #frob norm
                data.append((np.linalg.norm(noise), eps, tmp2))
    data = np.array(data).reshape(-1, len(noise_rate_list), num_random, 3).transpose(0,3,1,2)
    # with open(f'data/5PB_dim{dim_qudit}_noise_rate.pkl', 'wb') as fid:
    #     tmp0 = dict(data=data, num_random=num_random, dim_qudit=dim_qudit, noise_rate_list=noise_rate_list,
    #                 cvxpy_eps=cvxpy_eps, state_list=state_list, theta_optim=theta_optim)
    #     pickle.dump(tmp0, fid)


    fig,(ax0,ax1) = plt.subplots(1, 2, figsize=(8,4))
    for ind0 in range(data.shape[0]):
        tmp0= noise_rate_list
        tmp1 = data[ind0,1]
        ax0.fill_between(tmp0, tmp1.min(axis=1), tmp1.max(axis=1), alpha=0.2, color=cp_tableau[ind0])
        tmp2 = r'$\psi_-$' if ind0==0 else r'random $\sigma_'+f'{ind0}$'
        ax0.plot(tmp0, tmp1.mean(axis=1), color=cp_tableau[ind0], label=tmp2)

        tmp1 = data[ind0,2] / (data[ind0,0] + data[ind0,1])
        ax1.fill_between(tmp0, tmp1.min(axis=1), tmp1.max(axis=1), alpha=0.2, color=cp_tableau[ind0])
        ax1.plot(tmp0, tmp1.mean(axis=1), color=cp_tableau[ind0])
    ax0.set_ylabel(r'$\epsilon$')
    ax1.set_ylabel(r'$\frac{||Y-\sigma||_F}{\epsilon+||f||_2}$')
    fig.suptitle(f'5PB(d={dim_qudit}), 1/c={1/np.sqrt(theta_optim.fun):.1f}')
    ax0.legend()
    for ax in [ax0,ax1]:
        ax.set_xlabel(r'$||f||_2$')
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)
    # fig.savefig(f'data/5PB_dim{dim_qudit}_noise_rate.png', dpi=200)


def demo_UDA_worst_case():
    dim_qudit_list = [4,5,6,7,8,9]
    noise_rate = 1e-4
    num_random = 500
    cvxpy_eps = 1e-6

    dim_to_data_dict = dict()
    for dim_qudit in dim_qudit_list:
        print(dim_qudit)
        matrix_subspace = get_chebshev_orthonormal(dim_qudit, alpha=np.pi/dim_qudit, with_computational_basis=True)
        theta_optim,state_special = get_UDA_theta_optim_special_EVC(matrix_subspace, num_repeat=100, tol=1e-13, early_stop_threshold=1e-12)
        state_list = [state_special] + [rand_haar_state(dim_qudit) for _ in range(2)]
        dim_to_data_dict[(dim_qudit, 'theta_optim')] = theta_optim
        dim_to_data_dict[(dim_qudit, 'state_list')] = state_list

        data = []
        for state_i in state_list:
            measure_no_noise = ((matrix_subspace @ state_i) @ state_i.conj()).real
            for _ in tqdm(range(num_random)):
                tmp0 = np_rng.normal(size=len(matrix_subspace))
                noise = tmp0 * (noise_rate/np.linalg.norm(tmp0))
                tmp1,eps = density_matrix_recovery_SDP(matrix_subspace, measure_no_noise + noise, converge_eps=cvxpy_eps)
                tmp2 = np.linalg.norm(tmp1 - state_i[:,np.newaxis]*state_i.conj(), ord='fro') #frob norm
                tmp3 = get_fidelity(tmp1, state_i)
                data.append((eps, tmp2, tmp3))
        dim_to_data_dict[(dim_qudit, 'data')] = np.array(data).reshape(len(state_list), num_random, 3).transpose(0,2,1)
    # with open(f'data/5PB_worst_case_dim.pkl', 'wb') as fid:
    #     tmp0 = dict(dim_to_data_dict=dim_to_data_dict, dim_qudit_list=dim_qudit_list, noise_rate=noise_rate,
    #                 num_random=num_random, cvxpy_eps=cvxpy_eps)
    #     pickle.dump(tmp0, fid)

    ydata_list = [dim_to_data_dict[(x,'data')] for x in dim_qudit_list]
    fig,ax = plt.subplots()
    tmp0 = [1/np.sqrt(dim_to_data_dict[(x,'theta_optim')].fun) for x in dim_qudit_list]
    ax.plot(dim_qudit_list, tmp0, 'o-', color=cp_tableau[5], label='1/c')
    for ind0 in range(ydata_list[0].shape[0]):
        tmp0 = np.stack([x[ind0][1]/(x[ind0][0]+noise_rate) for x in ydata_list])
        ax.fill_between(dim_qudit_list, tmp0.min(axis=1), tmp0.max(axis=1), alpha=0.2, color=cp_tableau[ind0])
        tmp1 = r'$\psi_-$' if ind0==0 else r'random $\sigma_'+f'{ind0-1}$'
        ax.plot(dim_qudit_list, tmp0.mean(axis=1), 'o-', color=cp_tableau[ind0], label=tmp1)
    ax.set_xticks(dim_qudit_list)
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel(r'$d$')
    ax.set_ylabel(r'$\frac{||Y-\sigma||_F}{\epsilon+||f||_2}$')
    # ax.set_title(r'5PB worst case, $||f||_2=' + f'{noise_rate:.2g}$')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)
    # fig.savefig('data/5PB_worst_case_dim.png', dpi=200)

if __name__=='__main__':
    demo_search_UD_in_pauli_group_timing()
