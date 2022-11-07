# Quantum XXZ Chain

## Description

Exact diagonalization of arbitrary-S quntum XXZ chain. Hamiltonian is below:

$$
\begin{align}
    \hat{H} &= -J\sum_{\braket{i,j}}\left[ \hat{S}^x_i\hat{S}^x_j + \hat{S}^y_i\hat{S}^y_j+\Delta\hat{S}^z_i\hat{S}^z_j\right] \\
    &=-J\sum_{\braket{i,j}}\left[\frac{1}{2}\left(\hat{S}^+_i\hat{S}^-_j + \hat{S}^-_i\hat{S}^+_j\right) +\Delta\hat{S}^z_i\hat{S}^z_j\right]
\end{align}
$$

where $\braket{i,j}$ : coupling bond between $i$-th and $j$-th site.

## Install

```
git clone https://github.com/mktj2685/QuantumXXZChain.git
cd QuantumXXZChain
pip install -r requirements.txt
```

## Usage

### Arguments

```
--S (required): 
    Consider spin-S XXZ chain (e.g. 1/2, 1, 3/2, ...)

--N (required):
    Number of spins.

--Sz (optional, default is 0):
    Consider subspace whose specific Sz.

--J (optional, default is 1.0):
    Exchange coupling constant J.

--Delta (optional, default is 1.0):
    Anisotropic interaction parameter Δ.

--k (optional, default is 6):
    Number of eigenvalues and eigenvectors desired.
```

### Example

```
python xxz.py --S 1/2 --N 10 --Sz 0 --k 1
```

Result
```
E0 = -2.4999999999999893
```

This value consistent with the results of Bethe Ansatz ($E = -J\Delta\frac{N}{4}= -\frac{1}{4}$). 



## Reference

- https://youtu.be/Vwmd4LwvzUw?t=2748
- https://falexwolf.me/talks/2011-06-22-bethe.pdf
