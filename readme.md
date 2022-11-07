# Quantum XXZ Chain

## Description

Exact diagonalization of arbitrary-S quntum XXZ chain. Hamiltonian is below:

$$
    \hat{H} = \sum_{\braket{i,j}} J_{ij}(\hat{S}^x_i\hat{S}^x_j + \hat{S}^y_i\hat{S}^y_j+\Delta\hat{S}^z_i\hat{S}^z_j)
$$

where $\braket{i,j}$ : coupling bond between $i$-th and $j$-th site.

## Install

```
git clone https://github.com/mktj2685/QuantumXXZChain.git
cd QuantumXXZChain
pip install -r requirements.txt
```

## Usage

### Parameters

```
--S : 
    Consider spin-S XXZ chain (e.g. 1/2, 1, 3/2, ...)

--N :
    Number of spins.

--Sz :
    Consider subspace whose specific Sz.

--Delta :
    Anisotropic interaction parameter Δ.

--k :
    Number of eigenvalues and eigenvectors desired.
```

### Command

```
python xxz.py --S 3/2 --N 10 --Sz 0 --k 10
```