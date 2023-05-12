# README

This is the repository for the paper "Variational certifier for pure-state tomography".

This is a temporary repo and we are building on a powerful package to make it more user-friendly. please stay tuned.

Currently, this repo provides the following functions

1. determining whether a given set of measurement is UDA/UDP or not
2. searching for the optimal measurement for Pauli group, see `draft_uda_udp.py/demo_search_UD_in_pauli_group()`
3. the UDA/UDP minimum set over Pauli measurements
   * `data/pauli-indexB-core.json`
   * `pauli-indexB-full.json`: put in google drive (around 100 MB)
4. code to reproduce the figure/table in the paper `draft_paperfig.py`

```bash
conda create -y -n cuda118
conda install -y -n cuda118 -c conda-forge pytorch ipython pytest matplotlib scipy tqdm cvxpy
conda activate cuda118
pip install pyqet
```
