import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

cp_tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']


def demo_5PB_stability():
    with open(f'data/5PB_dim7_noise_rate.pkl', 'rb') as fid:
        data_figa = pickle.load(fid)
        # data num_random dim_qudit noise_rate_list cvxpy_eps state_list theta_optim

    with open(f'data/5PB_worst_case_dim.pkl', 'rb') as fid:
        data_figc = pickle.load(fid)
        # dim_to_data_dict dim_qudit_list noise_rate num_random cvxpy_eps

    # with open('data/20230307_1qudit_rand_orthonormal.pkl', 'rb') as fid:
    #     data_figd = pickle.load(fid)
    #     # exp_parameter_list hyper_parameter num_sample uda_data_list udp_data_list

    # with open('data/20230322_4PB_5PB_UDAP.pkl', 'rb') as fid:
    #     data_figd1 = pickle.load(fid)
    #     # hyperparameter dim_list uda_4pb_loss_list uda_5pb_loss_list udp_4pb_loss_list udp_5pb_loss_list

    fig,tmp0 = plt.subplots(2,2,figsize=(9,7))
    ax0,ax1,ax2,ax3 = tmp0[0][0],tmp0[0][1],tmp0[1][0],tmp0[1][1]
    FONTSIZE = 20

    for ax in [ax0,ax1]:
        ax.set_xlabel(r'$\lVert f\rVert_2$', fontsize=FONTSIZE)
        ax.set_xscale('log')
        ax.set_yscale('log')
    data = data_figa['data']
    marker_list = ['*', '1', '2'] #https://matplotlib.org/stable/gallery/lines_bars_and_markers/marker_reference.html
    for ind0 in range(data.shape[0]):
        tmp0= data_figa['noise_rate_list']
        tmp1 = data[ind0,1]
        ax0.fill_between(tmp0, tmp1.min(axis=1), tmp1.max(axis=1), alpha=0.2, color=cp_tableau[ind0])
        # tmp1u,tmp1m,tmp1l = tmp1.max(axis=1), tmp1.mean(axis=1), tmp1.min(axis=1)
        # ax0.errorbar(tmp0, tmp1m, yerr=[tmp1m-tmp1l,tmp1u-tmp1m], color=cp_tableau[ind0], alpha=0.2)
        tmp2 = r'$|\psi_-\rangle$' if ind0==0 else r'$\sigma_'+f'{ind0}$'
        ax0.plot(tmp0, tmp1.mean(axis=1), color=cp_tableau[ind0], label=tmp2, marker=marker_list[ind0], markersize=12)

        tmp1 = data[ind0,2] / (data[ind0,0] + data[ind0,1])
        ax1.fill_between(tmp0, tmp1.min(axis=1), tmp1.max(axis=1), alpha=0.2, color=cp_tableau[ind0])
        ax1.plot(tmp0, tmp1.mean(axis=1), color=cp_tableau[ind0], label=tmp2, marker=marker_list[ind0], markersize=12)
    ax0.set_ylabel(r'$\lVert \mathcal{M}_\mathbf{A}(Y^*)-b \rVert_2$', fontsize=FONTSIZE)
    ax1.set_ylabel(r'stability coefficient', fontsize=FONTSIZE)
    ax0.text(7e-4, 6e-7, '(a)', horizontalalignment='center', verticalalignment='center', fontsize=FONTSIZE)
    ax1.text(1.3e-6, 0.35, '(b)', horizontalalignment='center', verticalalignment='center', fontsize=FONTSIZE)
    ax1.axhline(8260, color=cp_tableau[5], linestyle='--', label=r'$\mathcal{L}^{-1/2}$')
    ax0.legend(fontsize=FONTSIZE-4)
    ax0.tick_params(axis='both', which='major', labelsize=FONTSIZE-5)
    ax1.tick_params(axis='both', which='major', labelsize=FONTSIZE-5)
    ax1.legend(loc='upper left', fontsize=FONTSIZE-6)
    ax0.grid(axis='y')
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')
    ax0.set_ylim(3.84e-7, 1.3e-3)

    # ax1.plot(data_figd1['dim_list'], data_figd1['uda_5pb_loss_list'], color=cp_tableau[0], linestyle='solid', label='5PB UDA')
    # ax1.plot(data_figd1['dim_list'], data_figd1['udp_4pb_loss_list'], color=cp_tableau[1], linestyle='solid', label='4PB UDP')
    # ax1.set_ylabel(r'loss function $\mathcal{L}$')
    # ax1.set_xlabel('qudit $d$')
    # ax1.set_xticks(np.arange(3,10))
    # ax1.set_yscale('log')
    # ax1.legend()
    # # ax1.axhline(1e-9, linestyle='dashed', color=cp_tableau[2])
    # ax1.set_yticks([1e-9, 1e-6, 1e-3, 1])
    # ax1.set_ylim(5e-11, 2)
    # ax1.yaxis.tick_right()
    # ax1.yaxis.set_label_position('right')
    # ax1.grid(axis='y')
    # ax1.text(3, 2e-10, '(b)', horizontalalignment='center', verticalalignment='center')#, fontsize=16

    dim_qudit_list = data_figc['dim_qudit_list']
    dim_to_data_dict = data_figc['dim_to_data_dict']
    ydata_list = [dim_to_data_dict[(x,'data')][:,2] for x in dim_qudit_list] #fidelity
    for ind0 in range(ydata_list[0].shape[0]):
        tmp0 = np.stack([x[ind0] for x in ydata_list])
        ax3.fill_between(dim_qudit_list, tmp0.min(axis=1), tmp0.max(axis=1), alpha=0.2, color=cp_tableau[ind0])
        tmp1 = r'$|\psi_-\rangle$' if ind0==0 else r'$\sigma_'+f'{ind0-1}$'
        ax3.plot(dim_qudit_list, tmp0.mean(axis=1), '-', color=cp_tableau[ind0], label=tmp1, marker=marker_list[ind0], markersize=12)
    ax3.set_xlabel(r'qudit $d$', fontsize=FONTSIZE)
    ax3.set_ylabel(r'fidelity', fontsize=FONTSIZE)
    ax3.text(4.1, 0.3, '(d)', horizontalalignment='center', verticalalignment='center', fontsize=FONTSIZE)
    ax3.legend(loc='center left', fontsize=FONTSIZE-5)
    ax3.yaxis.tick_right()
    ax3.tick_params(axis='both', which='major', labelsize=FONTSIZE-5)
    ax3.yaxis.set_label_position('right')

    dim_qudit_list = data_figc['dim_qudit_list']
    dim_to_data_dict = data_figc['dim_to_data_dict']
    ydata_list = [dim_to_data_dict[(x,'data')] for x in dim_qudit_list]
    for ind0 in range(ydata_list[0].shape[0]):
        tmp0 = np.stack([x[ind0][1]/(x[ind0][0]+data_figc['noise_rate']) for x in ydata_list])
        ax2.fill_between(dim_qudit_list, tmp0.min(axis=1), tmp0.max(axis=1), alpha=0.2, color=cp_tableau[ind0])
        tmp1 = r'$|\psi_-\rangle$' if ind0==0 else r'$\sigma_'+f'{ind0-1}$'
        ax2.plot(dim_qudit_list, tmp0.mean(axis=1), '-', color=cp_tableau[ind0], label=tmp1, marker=marker_list[ind0], markersize=12)
    tmp0 = [1/np.sqrt(dim_to_data_dict[(x,'theta_optim')].fun) for x in dim_qudit_list]
    ax2.plot(dim_qudit_list, tmp0, 'o', linestyle='dashed', color=cp_tableau[5], label=r'$\mathcal{L}^{-1/2}$')
    ax2.set_xticks(dim_qudit_list)
    ax2.set_xlim(3.8, 9.25)
    ax2.set_ylim(0.1, 6e4)
    ax2.set_yscale('log')
    ax2.set_xlabel(r'qudit $d$', fontsize=FONTSIZE)
    ax2.set_ylabel(r'stability coefficient', fontsize=FONTSIZE)
    ax2.text(8.8, 0.2, '(c)', horizontalalignment='center', verticalalignment='center', fontsize=FONTSIZE)
    ax2.legend(loc='upper left', fontsize=FONTSIZE-6)
    ax2.tick_params(axis='both', which='major', labelsize=FONTSIZE-5)
    # ax2.yaxis.tick_right()
    # ax2.yaxis.set_label_position('right')
    # ax.set_title(r'5PaB worst case, $\lVert f\rVert_2=' + f'{noise_rate:.2g}$')


    # exp_parameter_list = data_figd['exp_parameter_list']
    # uda_data_list = data_figd['uda_data_list']
    # key_list = sorted({(x[0],x[1]) for x in exp_parameter_list})
    # uda_data_dict = dict()
    # for key_i in key_list:
    #     ind0 = [x for x,y in enumerate(exp_parameter_list) if (y[:2]==key_i)]
    #     tmp0 = np.array([exp_parameter_list[x][2] for x in ind0])
    #     uda_data_dict[key_i] = tmp0, uda_data_list[ind0].max(axis=1)

    # for key_i, color_i in zip([x for x in key_list if x[0]==1], cp_tableau):
    #     tmp0 = uda_data_dict[key_i]
    #     ax3.plot(tmp0[0], tmp0[1], label=f'd={key_i[1]}', color=color_i, linestyle='solid')
    # ax3.set_ylabel(r'loss function $\mathcal{L}$')
    # ax3.set_xlabel('#orthonormal')
    # ax3.set_xticks(np.arange(3,8))
    # ax3.set_yscale('log')
    # ax3.set_ylim(1.5e-8, 5)
    # ax3.axhline(1e-5, linestyle='dashed')
    # ax3.set_xlim(2.3, 7.2)
    # ax3.set_yticks([1e-7, 1e-5, 1e-4, 1e-2, 1e0])
    # ax3.legend()
    # ax3.text(2.5, 5.5e-8, '(d)', horizontalalignment='center', verticalalignment='center')#, fontsize=16
    # ax3.yaxis.tick_right()
    # ax3.yaxis.set_label_position('right')

    fig.tight_layout()
    # fig.savefig('tbd00.png', dpi=200)
    fig.savefig('data/5PB_worst_case.pdf')
    fig.savefig('data/5PB_worst_case.png', dpi=200)


def demo_pauli_stability():
    print('zcdebug')
    with open('data/pauli_qubit4_noise_rate.pkl', 'rb') as fid:
        data_figa = pickle.load(fid)
        # data num_random num_qubit noise_rate_list cvxpy_eps state_list theta_optim

    with open('data/pauli_worst_case_dim.pkl', 'rb') as fid:
        data_figc = pickle.load(fid)

    fig,tmp0 = plt.subplots(2,2,figsize=(9,7))
    ax0,ax1,ax2,ax3 = tmp0[0][0],tmp0[0][1],tmp0[1][0],tmp0[1][1]
    FONTSIZE = 20
    marker_list = ['*', '1', '2']

    data = data_figa['data']
    for ind0 in range(data.shape[0]):
        tmp0= data_figa['noise_rate_list']
        tmp1 = data[ind0,1]
        ax0.fill_between(tmp0, tmp1.min(axis=1), tmp1.max(axis=1), alpha=0.2, color=cp_tableau[ind0])
        tmp2 = r'$|\psi_-\rangle$' if ind0==0 else r'$\sigma_'+f'{ind0}$'
        ax0.plot(tmp0, tmp1.mean(axis=1), color=cp_tableau[ind0], label=tmp2, marker=marker_list[ind0], markersize=12)

        tmp1 = data[ind0,2] / (data[ind0,0] + data[ind0,1])
        ax1.fill_between(tmp0, tmp1.min(axis=1), tmp1.max(axis=1), alpha=0.2, color=cp_tableau[ind0])
        ax1.plot(tmp0, tmp1.mean(axis=1), color=cp_tableau[ind0], label=tmp2, marker=marker_list[ind0], markersize=12)
    ax0.set_ylabel(r'$\lVert \mathcal{M}_\mathbf{A}(Y^*)-b \rVert_2$', fontsize=FONTSIZE)
    ax1.set_ylabel(r'stability coefficient', fontsize=FONTSIZE)
    ax0.text(9.2e-4, 8e-7, '(a)', horizontalalignment='center', verticalalignment='center', fontsize=FONTSIZE)
    ax1.text(1.2e-6, 0.08, '(b)', horizontalalignment='center', verticalalignment='center', fontsize=FONTSIZE)
    ax1.axhline(1.8957, color=cp_tableau[5], linestyle='--', label=r'$\mathcal{L}^{-1/2}$')
    ax0.tick_params(axis='both', which='major', labelsize=FONTSIZE-5)
    ax1.tick_params(axis='both', which='major', labelsize=FONTSIZE-5)
    # 1/np.sqrt(data_figa['theta_optim'].fun)
    for ax in [ax0,ax1]:
        ax.set_xlabel(r'$\lVert f \rVert_2$', fontsize=FONTSIZE)
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax0.legend(fontsize=FONTSIZE-4)
    ax1.legend(loc='upper right', fontsize=FONTSIZE-6)
    ax0.grid(axis='y')
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')


    num_qubit_list = data_figc['num_qubit_list']
    num_qubit_to_data_dict = data_figc['num_qubit_to_data_dict']
    ydata_list = [num_qubit_to_data_dict[(x,'data')] for x in num_qubit_list]
    for ind0 in range(ydata_list[0].shape[0]):
        tmp0 = np.stack([x[ind0][1]/(x[ind0][0]+data_figc['noise_rate']) for x in ydata_list])
        ax2.fill_between(num_qubit_list, tmp0.min(axis=1), tmp0.max(axis=1), alpha=0.2, color=cp_tableau[ind0])
        tmp1 = r'$|\psi_-\rangle$' if ind0==0 else r'$\sigma_'+f'{ind0-1}$'
        ax2.plot(num_qubit_list, tmp0.mean(axis=1), '-', color=cp_tableau[ind0], label=tmp1, marker=marker_list[ind0], markersize=12)
    tmp0 = [1/np.sqrt(num_qubit_to_data_dict[(x,'theta_optim')].fun) for x in num_qubit_list]
    ax2.plot(num_qubit_list, tmp0, 'o', linestyle='dashed', color=cp_tableau[5], label=r'$\mathcal{L}^{-1/2}$')
    ax2.set_xticks(num_qubit_list)
    ax2.set_xlim(1.7, 5.3)
    # ax2.set_yscale('log')
    ax2.set_xlabel('No. of qubits', fontsize=FONTSIZE)
    ax2.set_ylabel(r'stability coefficient', fontsize=FONTSIZE)
    ax2.text(5.1, 0.4, '(c)', horizontalalignment='center', verticalalignment='center', fontsize=FONTSIZE)
    ax2.legend(loc='upper left', fontsize=FONTSIZE-7)
    ax2.tick_params(axis='both', which='major', labelsize=FONTSIZE-4)
    # ax2.yaxis.tick_right()
    # ax2.yaxis.set_label_position('right')
    # ax.set_title(r'5PaB worst case, $\lVert f\rVert_2=' + f'{noise_rate:.2g}$')

    num_qubit_list = data_figc['num_qubit_list']
    num_qubit_to_data_dict = data_figc['num_qubit_to_data_dict']
    ydata_list = [np.minimum(1,num_qubit_to_data_dict[(x,'data')][:,2]) for x in num_qubit_list] #fidelity
    for ind0 in range(ydata_list[0].shape[0]):
        tmp0 = np.stack([x[ind0] for x in ydata_list])
        ax3.fill_between(num_qubit_list, tmp0.min(axis=1), tmp0.max(axis=1), alpha=0.2, color=cp_tableau[ind0])
        tmp1 = r'$|\psi_-\rangle$' if ind0==0 else r'$\sigma_'+f'{ind0-1}$'
        ax3.plot(num_qubit_list, tmp0.mean(axis=1), '-', color=cp_tableau[ind0], label=tmp1, marker=marker_list[ind0], markersize=12)
    ax3.set_xlabel('No. of qubits', fontsize=FONTSIZE)
    ax3.set_ylabel(r'fidelity', fontsize=FONTSIZE)
    ax3.text(2.2, 0.999958, '(d)', horizontalalignment='center', verticalalignment='center', fontsize=FONTSIZE)
    ax3.legend(loc='lower right', fontsize=FONTSIZE-6)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position('right')
    ax3.tick_params(axis='both', which='major', labelsize=FONTSIZE-5)

    fig.tight_layout()
    fig.savefig('data/pauli_worst_case.png', dpi=200)
    fig.savefig('data/pauli_worst_case.pdf')

def demo_UDA_UDP_eigenvalue():
    hf_hex_to_rgb = lambda h: (int(h[1:3],16)/255, int(h[3:5],16)/255, int(h[5:7],16)/255)
    tmp0 = {'zero': cp_tableau[3], 'non-trace0': cp_tableau[6],
            'loss': cp_tableau[1], 'valid': cp_tableau[4], 'line':cp_tableau[9]}
    key_to_color = {k:hf_hex_to_rgb(v) for k,v in tmp0.items()}

    dim = 7
    image_uda = np.ones((dim+1,dim+1,3), dtype=np.float64)
    image_uda[0,0] = key_to_color['zero'] #black
    image_uda[0,1:] = key_to_color['non-trace0']
    image_uda[1:,0] = key_to_color['non-trace0']
    image_uda[1,1:-1] = key_to_color['loss']
    image_uda[1:-1,1] = key_to_color['loss']
    for x in range(2,dim+1):
        image_uda[x,2:(dim+1-x)] = key_to_color['valid']

    image_udp = image_uda.copy()
    image_udp[0,1:3] = key_to_color['non-trace0']
    image_udp[1:3,0] = key_to_color['non-trace0']
    image_udp[1,1] = key_to_color['loss']
    for x in range(dim+1):
        image_udp[x,max(0,3-x):(dim+1-x)] = key_to_color['valid']

    fig,(ax0,ax1) = plt.subplots(1, 2, figsize=(8,4))
    ax0.imshow(image_uda, origin='lower')
    ax0.plot([-0.5,-0.5], [-0.5,dim+0.5], color=key_to_color['line'])
    ax0.plot([-0.5,dim+0.5], [-0.5,-0.5], color=key_to_color['line'])
    for x in range(dim+1):
        ax0.plot([x+0.5,x+0.5], [-0.5,dim+0.5-x], color=key_to_color['line'])
        ax0.plot([-0.5,dim+0.5-x], [x+0.5,x+0.5], color=key_to_color['line'])
    ax0.fill_between([-3,-2], [0,0], [0.5,0.5], color=key_to_color['non-trace0'], label='not traceless')
    ax0.fill_between([-3,-2], [0,0], [0.5,0.5], color=key_to_color['loss'], label='loss function')
    ax0.fill_between([-3,-2], [0,0], [0.5,0.5], color=key_to_color['valid'], label='UDA/UDP')

    ax1.imshow(image_udp, origin='lower')
    ax1.plot([-0.5,-0.5], [-0.5,dim+0.5], color=key_to_color['line'])
    ax1.plot([-0.5,dim+0.5], [-0.5,-0.5], color=key_to_color['line'])
    for x in range(dim+1):
        ax1.plot([x+0.5,x+0.5], [-0.5,dim+0.5-x], color=key_to_color['line'])
        ax1.plot([-0.5,dim+0.5-x], [x+0.5,x+0.5], color=key_to_color['line'])
    for ax in [ax0,ax1]:
        ax.set_xlabel('#positive eigenvalues $n_+$')
        ax.set_ylabel('#negative eigenvalues $n_-$')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(-0.5, dim+0.5)
        ax.set_ylim(-0.5, dim+0.5)
    ax0.legend()
    fig.tight_layout()
    # fig.savefig('data/uda_udp_eigenvalue.png', dpi=200)


def compare_3PB_4PB_5PB():
    with open('data/20230322_4PB_5PB_UDAP.pkl', 'rb') as fid:
        tmp0 = pickle.load(fid)
        dim_list = tmp0['dim_list']
        udp_3pb_loss_list = tmp0['udp_3pb_loss_list']
        udp_4pb_loss_list = tmp0['udp_4pb_loss_list']
        uda_5pb_loss_list = tmp0['uda_5pb_loss_list']
        uda_4pb_loss_list = tmp0['uda_4pb_loss_list']
        # hyperparameter dim_list uda_4pb_loss_list uda_5pb_loss_list udp_4pb_loss_list udp_5pb_loss_list

    fig,(ax0,ax1) = plt.subplots(1,2,figsize=(8,4))
    ax0.plot(dim_list, udp_4pb_loss_list, label='4PBs', marker='.', markersize=8)
    ax0.plot(dim_list, udp_3pb_loss_list, label='3PBs', marker='x', markersize=8)
    ax1.plot(dim_list, uda_5pb_loss_list, label='5PBs', marker='.', markersize=8)
    ax1.plot(dim_list, uda_4pb_loss_list, label='4PBs', marker='x', markersize=8)
    for ax in [ax0,ax1]:
        ax.set_xlabel('qudit $d$', fontsize=20)
        ax.legend(fontsize=18)
        ax.set_yscale('log')
        ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylim(3e-14, 0.4)
    ax0.set_ylabel(r'UDP loss function', fontsize=20)
    ax1.set_ylabel(r'UDA loss function', fontsize=20)
    ax0.text(8.8, 2e-13, '(a)', horizontalalignment='center', verticalalignment='center', fontsize=18)
    ax1.text(8.8, 2e-13, '(b)', horizontalalignment='center', verticalalignment='center', fontsize=18)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')
    fig.tight_layout()
    # fig.savefig('tbd00.png', dpi=200)
    fig.savefig('data/20230511_3PB_4PB_5PB_UDAP.pdf')
    fig.savefig('data/20230511_3PB_4PB_5PB_UDAP.png', dpi=200)


if __name__=='__main__':
    demo_5PB_stability()
    demo_pauli_stability()
    # compare_3PB_4PB_5PB()
