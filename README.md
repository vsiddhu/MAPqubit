# MAPQubit

This is a code for reproducing the results in a manuscript titled,
[Maximum a posteriori estimation of quantum states](https://arxiv.org/abs/1805.12235).

### Prerequisites

- Python 2.7
- ipython 2.4.1
- scipy==1.1.0
- numpy==1.15.4
- astropy==1.3
- Maybe more, use `pip install` if you have issues.

## Running simulation

If you run 

'''
python expPar.py
'''

without changing any parameters, the code will generate measurement data, 
perform MAP and MLE reconstruction and save data in multiple files. If one uses 
40 cores (change nProc=4 to nProc=40 in expPar.py) on a AMD Opteron(tm) 
Processor 6376 with 128 GB RAM to run this command it takes about 2 hours.
 

## Generating plots

In the 'data' folder, where simulation results from running 'expPar.py' have
already been stored, with the aid of ipython notebooks 'fidPlotN.ipynb' and 'l1DPlotN.ipynb'
one can generate various plots, some of which are included in the manuscript.

## Tutorial: Convex Optimization over qubits 

In the 'prGD' folder, a the notebook titled 'gradDesc.ipynb' has a simple 
example that shows how minimization of a convex function over qubit density 
operators can be done using projected gradient descent. In this code one
also computes the surrogate duality gap.

## Contributors
If you end up using this code, please cite the following paper - 

Vikesh Siddhu, Maximum a posteriori probability estimates for quantum tomography, Phys. Rev. A 99, 012342 (2019) 

```
@article{PhysRevA.99.012342,
  title = {Maximum a posteriori probability estimates for quantum tomography},
  author = {Siddhu, Vikesh},
  journal = {Phys. Rev. A},
  volume = {99},
  issue = {1},
  pages = {012342},
  numpages = {7},
  year = {2019},
  month = {Jan},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevA.99.012342},
  url = {https://link.aps.org/doi/10.1103/PhysRevA.99.012342}
}
```

Report bugs and missing info to vsiddhuATandrewDOTcmuDOTedu (replace AT, DOT appropriately).

## License

This program is free software; you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation - see the [LICENSE](LICENSE) file for details
