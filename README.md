# README

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7995116.svg)](https://zenodo.org/doi/10.5281/zenodo.7995115)

This is the repository for the paper "A Variational Approach to Unique Determinedness in Pure-state Tomography". [arxiv-link](https://arxiv.org/abs/2305.10811)

🚀 Exciting News! We've launched the `numqi` package [github/numqi](https://github.com/husisy/numqi), combining all the functionalities of this repository and even more! 🌟 To dive into these features, just install `numqi` using `pip install numqi`, and explore the relevant functions within the `numqi.unique_determine` module. 🛠️

Currently, this repo provides the following functions

1. determining whether a given set of measurement is UDA/UDP or not
2. searching for the optimal measurement for Pauli group, see `draft_uda_udp.py/demo_search_UD_in_pauli_group()`
3. the UDA/UDP minimum set over Pauli measurements
   * `data/pauli-indexB-core.json`
   * `pauli-indexB-full.json`: [google-drive-link](https://drive.google.com/file/d/138XlVUSgOYXh7VqENPgRgsH9UcilVMob/view?usp=sharing) (around 200 MB)
4. code to reproduce the figure/table in the paper `draft_paperfig.py`

```bash
conda create -y -n cuda118
conda install -y -n cuda118 -c conda-forge pytorch ipython pytest matplotlib scipy tqdm cvxpy
conda activate cuda118
```

quickstart

```Python
from draft_uda_udp import demo_pauli_loss_function
demo_pauli_loss_function() #takes around several hours
```

```Python
# search UD in Pauli groups, default parameters are for 3-qubits, it takes several minutes for one cpu core to run one search
from draft_uda_udp import demo_search_UD_in_pauli_group
demo_search_UD_in_pauli_group()
```

Every function with prefix `demo_` should be runable. The functions in `draft_paperfig.py` are to generate figures used in the paper.
