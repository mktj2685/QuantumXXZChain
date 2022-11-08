import argparse
from typing import List, Tuple
from fractions import Fraction
from itertools import product
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from memory_profiler import profile

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--S', type=str, help='Consider spin-S chain (e.g. 1/2, 1, 3/2, ...).')
    parser.add_argument('--N', type=int, help='Number of spins.')
    parser.add_argument('--Sz', type=str, default='0', help='Consider specific Sz subspace.')
    parser.add_argument('--J', type=float, default=1.0, help='Exchange coupling constant.')
    parser.add_argument('--Delta', type=float, default=1.0, help='XXZ anisotropic interaction parameter.')
    parser.add_argument('--k', type=int, default=6, help='Number of eigenvalues and eigenvectors desired.')
    parser.add_argument('--sigma', type=float, default=10.0, help='Shift-invert mode parameter.')
    return parser.parse_args()

@profile
def basis(S:Fraction, Sz:Fraction, N:int) -> List[Tuple[Fraction]]:
    m = [S-i for i in range(int(2*S+1))]                    # Get possible Sz values.
    states = product(m, repeat=N)                           # Get all possible states. Number of states is (2S+1)^n.
    states = [state for state in states if sum(state)==Sz]  # remain states which total Sz is Sz.
    return states

@profile
def hamiltonian(S:Fraction, Sz:Fraction, N:int, bonds:List[Tuple], J:float, Delta:float) -> np.ndarray:
    maxSz = S                                           # max Sz (Fraction)
    minSz = -S                                          # min Sz (Fraction)
    states = basis(S, Sz, N)                    # index (int) -> state (List[Tuple[Fraction]])
    indices = dict(zip(states, range(len(states))))     # state (List[Tuple[Fraction]]) -> index (int)
    dim = len(states)                                   # Hamiltonian dimension
    H = np.zeros((dim, dim), dtype=np.float64)          # Hamiltonian matrix

    # Calculate Hamiltonian elements.
    for state in states:
        for bond in bonds:
            i, j = bond
            idx = indices[state]
            Szi = float(state[i])                           # i-th site Sz value.
            Szj = float(state[j])                           # j-th site Sz value.
            H[idx, idx] += -1.0 * J * Delta * Szi * Szj     # diagonal term SziSzj.
            if state[i] != maxSz and state[j] != minSz:     # off-diagonal term S+iS-j/2.
                state_ = list(state)
                state_[i] += 1
                state_[j] -= 1
                idx_ = indices[tuple(state_)]
                H[idx, idx_] += -0.5 * J
            if state[i] != minSz and state[j] != maxSz:     # off-diagonal term S-iS+j/2.
                state_ = list(state)
                state_[i] -= 1
                state_[j] += 1
                idx_ = indices[tuple(state_)]
                H[idx, idx_] += -0.5 * J
    return H

@profile
def hamiltonian_csr(S:Fraction, Sz:Fraction, N:int, bonds:List[Tuple], J:float, Delta:float, fmt:str='csr'):
    maxSz = S                                           # max Sz (Fraction)
    minSz = -S                                          # min Sz (Fraction)
    states = basis(S, Sz, N)                            # index (int) -> state (List[Tuple[Fraction]])
    indices = dict(zip(states, range(len(states))))     # state (List[Tuple[Fraction]]) -> index (int)
    dim = len(states)                                   # Hamiltonian dimension
    data = []                                           # non-zero elements
    row = []                                            # row of non-zero elements.
    col = []                                            # col of non-zero elements.

    # Calculate Hamiltonian elements.
    for state in states:
        for bond in bonds:
            i, j = bond
            idx = indices[state]
            Szi = float(state[i])                           # i-th site Sz value.
            Szj = float(state[j])  
            diag = -1.0 * J * Delta * Szi * Szj
            data.append(diag)
            row.append(idx)
            col.append(idx)
            if state[i] != maxSz and state[j] != minSz:     # off-diagonal term S+iS-j/2.
                state_ = list(state)
                state_[i] += 1
                state_[j] -= 1
                idx_ = indices[tuple(state_)]
                offdiag = -0.5 * J
                data.append(offdiag)
                row.append(idx)
                col.append(idx_)
            if state[i] != minSz and state[j] != maxSz:     # off-diagonal term S-iS+j/2.
                state_ = list(state)
                state_[i] -= 1
                state_[j] += 1
                idx_ = indices[tuple(state_)]
                offdiag = -0.5 * J
                data.append(offdiag)
                row.append(idx)
                col.append(idx_)
    return csr_matrix((data, (row, col)), (dim, dim))

if __name__ == '__main__':
    # Parse arguments.
    args = parse_args()
    S = Fraction(args.S)
    Sz = Fraction(args.Sz)
    # bonds = [(i, i+1) for i in range(N-1)]                # OBC
    bonds = [(i, i+1) for i in range(args.N-1)] + [(args.N-1, 0)]     # PBC

    # Get Hamiltonian matrix.
    # H = hamiltonian(S, Sz, args.N, bonds, args.J, args.Delta)
    H = hamiltonian_csr(S, Sz, args.N, bonds, args.J, args.Delta)
    dim = H.shape[0]

    # Diagonalize.
    k = args.k if args.k < dim else dim
    # NOTE see https://stackoverflow.com/questions/12125952/scipys-sparse-eigsh-for-small-eigenvalues
    w, v = eigsh(H, k=k, which='LA', mode='normal', sigma=args.sigma)
    for i in range(k):
        print(f'E{i} = {w[i]}')
        print(f'|Ψ{i}> = {v[:,i]}')
