import math
import numpy as np
import pandas as pd
import PIL.Image
import matplotlib.pyplot as plt

import streamlit as st
from trubrics.integrations.streamlit import FeedbackCollector

from utils import rand_haar_state, save_index_to_file, get_pauli_group
from ud_utils import density_matrix_recovery_SDP

def is_valid_int_str(x):
    try:
        int(x)
        ret = True
    except:
        ret = False
    return ret

@st.cache_data
def load_pauli_UD(num_qubit, seed):
    all_pauli_str = get_pauli_group(num_qubit, kind='str')
    index_list = save_index_to_file('data/pauli-indexB-core.json', str(num_qubit))
    len_min = min(len(x) for x in index_list)
    index_list = [x for x in index_list if len(x)==len_min]
    np_rng = np.random.default_rng(seed)
    pauli_index = index_list[np_rng.integers(0, len(index_list))]
    pauli_str = [all_pauli_str[x] for x in pauli_index]
    tmp0 = get_pauli_group(num_qubit, kind='numpy', use_sparse=True)
    pauli_np = [tmp0[x] for x in pauli_index]
    return pauli_index,pauli_str,pauli_np

def print_pauli_str(pauli_str, N0=11):
    tmp1 = math.ceil(len(pauli_str)/N0)
    z0 = [','.join(r'\mathrm{'+y+'}' for y in pauli_str[i*N0:(i+1)*N0]) for i in range(tmp1)]
    z0[0] = r'\{A\}=\{' + z0[0]
    z0[-1] = z0[-1] + r'\}'
    for x in z0:
        st.latex(x)

def st_table_numpy(np0, N0, round_digit=3):
    tmp0 = [str(x).strip('()') for x in np.round(np0,round_digit).reshape(-1).tolist()]
    tmp0 += [''] * (math.ceil(len(tmp0)/N0)*N0 - len(tmp0))
    tmp1 = np.array(tmp0).reshape(-1, N0)
    st.table(tmp1)

def button_to_randomize_seed(text):
    # def hf0():
    #     del st.session_state['seed']
    button = st.button(text)#, on_click=hf0)
    if button and ('seed' in st.session_state):
        del st.session_state['seed']

st.title('Interactive demo for UDAP')

st.markdown('A Variational Approach to Unique Determinedness in Pure-state Tomography [arxiv-link](https://arxiv.org/abs/2305.10811)')
st.markdown('open source code [github-link](https://github.com/husisy/udap-public)')

st.markdown('This page will show:')
st.markdown('1. what is pure state tomography')
st.markdown('2. what is the difference between UDA and UDP')
st.markdown('3. Our variational method and results')

# st.subheader('Hyperparameters')
# st.markdown('You may continue with these default values, and go back to change them to see different results later.')

option = st.selectbox('number of qubits to play with?', ('2 qubits', '3 qubits', '4 qubits'))
num_qubit = int(option[0])
# num_qubit = st.slider('number of qubits $n$=', min_value=2, max_value=4)

tmp0 = '**Random seed** (input empty to randomize)'
seed = st.session_state.get('seed', None)
if (seed is None) or (not is_valid_int_str(seed)):
    seed = int(np.random.default_rng().integers(int(1e18)))
else:
    seed = int(seed)
np_rng = np.random.default_rng(seed)
st.text_input(tmp0, str(seed), key='seed')


st.subheader('Random state')
st.markdown('set different random seed to generate different random pure state')
q0 = rand_haar_state(2**num_qubit, seed=np_rng)
st.markdown(rf'{num_qubit}-qubits pure state $|\psi\rangle$')
tmp0 = (r'\\').join(str(x).strip('()') for x in np.round(q0,3).tolist())
st.latex(r'|\psi\rangle=\begin{bmatrix}'+ tmp0 + r'\end{bmatrix}')
# st_table_numpy(q0, 1)
button_to_randomize_seed('re-generate state')


st.subheader('Pure state tomography')

st.markdown('For $n$-qubits density matrix, we need measure $4^n-1$ Pauli operators to reconstruct it (tomography). '
            'If we know the state is pure, a smaller set of measurements is enough.')

st.markdown('Previous research [XianMa2016] proved that, $2$-qubits pure state needs at least $11$ Pauli operators (include $I$), '
            'and $3$-qubits pure state needs at least $31$ Pauli operators (include $I$).')

UD_pauli_index,UD_pauli_str,UD_pauli_np = load_pauli_UD(num_qubit, seed)
st.markdown(r'Below we choose one set of these Pauli operators $\{A\}$ to do pure state tomography.'
            ' (not unique, you can change random seed to get differnt set)')
print_pauli_str(UD_pauli_str)
button_to_randomize_seed('re-generate Pauli operators')

st.markdown(r'To measure these operators, simply calculate $a_i=\langle\psi|A_i|\psi\rangle+f$ for each $A_i\in\{A\}$.'
            'where $f$ is the random measurement error chosen from uniform distribution.')
checkbox = st.checkbox(r"click here to perform measurement")
if checkbox:
    tmp0 = st.slider(r'measurement error rate in log-scale $\log(f)$=', min_value=-3.0, max_value=-0.0, value=-2.0, step=0.1)
    noise_rate = 10**tmp0
    tmp1 = round(noise_rate, 3)
    noise_f = np_rng.uniform(-noise_rate,noise_rate,size=len(UD_pauli_np))
    UD_pauli_measure = np.array([np.vdot(q0, x@q0) for x in UD_pauli_np]).real

    fig,ax = plt.subplots()
    np0 = UD_pauli_measure
    np1 = UD_pauli_measure + noise_f
    N0 = np0.shape[0]
    width = 1 / (N0*2+1)
    x0 = (np.arange(N0)+0.5)/N0
    ax.bar(x0-width/2, np0, width, label='without noise')
    ax.bar(x0+width/2, np1, width, label='with noise')
    if num_qubit==2:
        ax.set_xticks(x0, UD_pauli_str)
    else:
        ax.set_xticks([])
        ax.set_xlabel('Pauli operators')
    ax.set_ylabel(r'measurement $a_i$')
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    # st.markdown(r'$a_i=\langle\psi|A_i|\psi\rangle$=')
    # st_table_numpy(UD_pauli_measure, N0=6)

    st.markdown('Given above measurement results (a bunch of numbers $a_i$), we can recover the state by solving a semi-definite programming.')
    st.latex(r'\mathrm{find}\;\rho')
    st.latex(r'''\mathrm{s.t.}\;\begin{cases} \mathrm{Tr}\left[\rho A_{i}\right]=a_{i}\\ \rho\succeq 0 \end{cases}''')
    st.caption('This is the UDA case and will give a **unique** solution, more details see next section.')

    checkbox = st.checkbox(r"click here to perform state recovery")
    if checkbox:
        tmp0 = np.stack([x.toarray() for x in UD_pauli_np])
        rho_recover,_ = density_matrix_recovery_SDP(tmp0, UD_pauli_measure+noise_f, converge_eps=1e-4)
        if num_qubit==2:
            st.markdown(r'the restored state $\rho=$')
            st_table_numpy(rho_recover, N0=4, round_digit=3)
        else:
            st.markdown(r'the density matrix $\rho$ is too large to show, you can check the code to see the result')
        st.markdown(r"Let's calculate the fidelity between the recovered state and the original state")
        fidelity = np.vdot(q0, rho_recover @ q0).real
        st.latex(rf'\langle\psi|\rho|\psi\rangle={fidelity:.3f}')
        st.markdown('If the fidelity is 1, which should always be the case without noise $f$, the recovered state is exactly the original state. ')

st.subheader('What is UDA and UDP')
checkbox = st.checkbox('click here to expand', key='uda-udp-diff')
if checkbox:
    st.markdown('UDAP is short for Unique Determinedness over All states (UDA) and Unique Determinedness over all Pure states (UDP)'
                'The example in previous section is a UDA case, because the optimization is searching among all density matrix. '
                'However, the optimization problem for UDP is searching among all pure states as follow.')
    st.latex(r'\mathrm{find}\;|\phi\rangle')
    st.latex(r'''\mathrm{s.t.}\;\langle\phi| A_{i}|\phi\rangle =a_{i}''')
    st.markdown(r'If the observable set \{A\} is UDP, then the solution is **unique**.')
    st.markdown(r'Apparently, all UDA observables \{A\} are also UDP observables, but not vice versa. '
                'An example from [ClaudioCarmeli2015]: the four polynomial bases (4PB) is UDP but not UDA. '
                'the five Pauli bases (5PB) is UDA and also UDP. Please check paper for the explicit formula.')

st.subheader('Our variational method and results')

st.markdown("It's natural to ask, how to find a set of UDA/UDP observables? Study [ClaudioCarmeli2014] "
            "give a necessary and sufficient condition for a set of observables to be UDA/UDP. ")
checkbox = st.checkbox('click here to expand', key='uda-udp-iff')
if checkbox:
    st.markdown(r"> Given a set of observables $\{A\}$, for all nonzero matrix $x$ in its orthogaonal space $\{A\}^\perp$, "
                r"let $n_+/n_-$ be the number of strictly positive and negative eigenvalues of $x$, then "
                r"$\{A\}$ is UDA if and only if")
    st.latex(r"min(n_-,n_+)\geq 2")
    st.markdown(r"> and $\{A\}$ is UDP if and only if")
    st.latex(r"max(n_-,n_+)\geq 2")

st.markdown("Based on this condition, we propose a variational method to search UDA/UDP. ")
checkbox = st.checkbox('click here to expand', key='uda-udp-variational')
if checkbox:
    st.markdown('To parameterize all matrix that violating the UDP condition,')
    st.latex(r'\Delta(\vec{\lambda},\vec{\psi}) = \lambda_{1}|\psi_{1}\rangle\langle\psi_{1}|+\lambda_{2}|\psi_{2}\rangle\langle\psi_{2}|')
    st.markdown('similarly for UDA,')
    st.latex(r'\Delta(\vec{\lambda},\vec{\psi}) = -\lambda_{1}|\psi_{1}\rangle\langle\psi_{1}|+\sum_{i=2}^{d}\lambda_{i}|\psi_{i}\rangle\langle\psi_{i}|')
    st.markdown(r'where $\lambda$ are positive numbers, and $\psi$ are orthonormal vectors. Then the loss function $\mathcal{L}$ is defined as')
    st.latex(r'\mathcal{M}_{\mathbf{A}}[\rho]=\left(\text{Tr}[A_0\rho], \text{Tr}[A_1\rho],...,\text{Tr}[A_m\rho]\right)')
    st.latex(r'\mathcal{L}_{\mathbf{A}}(\vec{\lambda},\vec{\psi})=\left\Vert \mathcal{M}_{\mathbf{A}}\left[\Delta(\vec{\lambda},\vec{\psi})\right]\right\Vert _{2}^{2}')
st.markdown(r'If the optimized loss function $\min\mathcal{L}$ is zero, then the observable set $\{A\}$ is not UDA/UDP. '
            r'If the optimized loss function $\min\mathcal{L}$ is nonzero, then the observable set $\{A\}$ is likely to be UDA/UDP. '
            'Then, We can search UDA/UDP observable sets for 2,3,4,5 qubits **numerically**. ')
tmp0 = {
    '#qubits': [2, 3, 4, 5],
    'UDA loss': [1, 0.519, 0.280, 0.202],
    'UDP loss': [2,2,1.788,1.951],
    '#Pauli': ['11,13', '31,32,34','106,107,108', '393,395,397'],
}
pd0 = pd.DataFrame({k:[str(x) for x in v] for k,v in tmp0.items()})
st.table(pd0)
st.markdown(r'We reproduce the minimum set $11/31$ for 2/3 qubits (in less than several minutes on laptop),'
            r' and find the possible minimum number of new UDA/UDP sets for 4/5 qubits is $106/393$ respectively.')
st.caption('Due to the limitation of computational resources, the searching algorithm cannot run on the website. '
           'Please download code from [github-link](https://github.com/husisy/udap-public) and run locally.')

st.markdown('Moreover, we can randomly sample Pauli sets and check whether they are UDA/UDP. ')
tmp0 = PIL.Image.open('data/20230309_n_pauli_sucess_probability.png')
st.image(tmp0, caption='the probability for randomly-sampled Pauli sets to be UDA/UDP')

st.markdown('There are more interesting results about UDA/UDP topic,')
st.markdown('1. the connection between the state recovery stability and the optimized loss value and '
            'the effect of the noise in the state recovery. ')
st.markdown('2. (unsolved) our numerical results show that all UDP pauli set are also UDA, but we cannot prove it. ')
st.markdown('please check our paper for more details. ')

st.subheader('References')

st.markdown('[XianMa2016] Pure-state tomography with the expectation value of Pauli operators '
            '[https://doi.org/10.1103/PhysRevA.93.032140](https://doi.org/10.1103/PhysRevA.93.032140)')
st.markdown('[ClaudioCarmeli2014] Tasks and premises in quantum state determination'
            '[https://doi.org/10.1088/1751-8113/47/7/075302](https://doi.org/10.1088/1751-8113/47/7/075302)')
st.markdown('[ClaudioCarmeli2015] How many orthonormal bases are needed to distinguish all pure quantum states? '
            '[https://doi.org/10.1140/epjd/e2015-60230-5](https://doi.org/10.1140/epjd/e2015-60230-5)')

st.subheader('Feedback')

collector = FeedbackCollector(
    component_name="udap-public",
    email=st.secrets.trubrics.email, #.streamlit/secrets.toml
    password=st.secrets.trubrics.password,
)

collector.st_feedback(
    feedback_type="thumbs",
    model='none',
    # metadata=dict(seed=seed, text=text_input),
    open_feedback_label="[Optional] Provide additional feedback",
)
