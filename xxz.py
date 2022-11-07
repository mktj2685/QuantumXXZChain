import argparse
from typing import List, Tuple
from fractions import Fraction
from itertools import product
import numpy as np
from scipy.sparse.linalg import eigsh

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--S', type=str, help='Consider spin-S chain (e.g. 1/2, 1, 3/2, ...).')
    parser.add_argument('--N', type=int, help='Number of spins.')
    parser.add_argument('--Sz', type=str, default='0', help='Consider specific Sz subspace.')
    parser.add_argument('--Delta', type=float, default=1.0, help='XXZ anisotropic interaction parameter.')
    parser.add_argument('--k', type=int, default=6, help='Number of eigenvalues and eigenvectors desired.')
    return parser.parse_args()

def get_states_Sz(S:Fraction, Sz:Fraction, N:int) -> List[Tuple[Fraction]]:
    m = [S-i for i in range(int(2*S+1))]                    # Get possible Sz values.
    states = product(m, repeat=N)                           # Get all possible states. Number of states is (2S+1)^n.
    states = [state for state in states if sum(state)==Sz]  # remain states which total Sz is Sz.
    return states

def hamiltoniah(S:Fraction, Sz:Fraction, N:int, bonds:List[Tuple], J:List[float], Delta:float) -> np.ndarray:
    maxSz = S                                           # max Sz (Fraction)
    minSz = -S                                          # min Sz (Fraction)
    states = get_states_Sz(S, Sz, N)                    # index (int) -> state (List[Tuple[Fraction]])
    indices = dict(zip(states, range(len(states))))     # state (List[Tuple[Fraction]]) -> index (int)
    dim = len(states)                                   # Hamiltonian dimension
    H = np.zeros((dim, dim), dtype=np.float64)          # Hamiltonian matrix

    # Calculate Hamiltonian elements.
    for state in states:
        for bond, Jij in zip(bonds, J):
            i, j = bond
            idx = indices[state]
            Szi = float(state[i])                           # i-th site Sz value.
            Szj = float(state[j])                           # j-th site Sz value.
            H[idx, idx] += Jij * Delta * Szi * Szj          # diagonal term SziSzj.
            if state[i] != maxSz and state[j] != minSz:     # off-diagonal term S+iS-j/2.
                state_ = list(state)
                state_[i] += 1
                state_[j] -= 1
                idx_ = indices[tuple(state_)]
                H[idx, idx_] += 0.5 * Jij
            if state[i] != minSz and state[j] != maxSz:     # off-diagonal term S-iS+j/2.
                state_ = list(state)
                state_[i] -= 1
                state_[j] += 1
                idx_ = indices[tuple(state_)]
                H[idx, idx_] += 0.5 * Jij
    return H

if __name__ == '__main__':
    # Set experiment configures.
    args = parse_args()
    N = args.N
    S = Fraction(args.S)
    Sz = Fraction(args.Sz)
    Delta = args.Delta
    # bonds = [(i, i+1) for i in range(N-1)]                # OBC
    bonds = [(i, i+1) for i in range(N-1)] + [(N-1, 0)]     # PBC
    J = [1.0 for i in range(len(bonds))]

    # Get Hamiltonian matrix.
    H = hamiltoniah(S, Sz, N, bonds, J, Delta)
    dim = H.shape[0]

    # Diagonalize.
    k = args.k if args.k < dim else dim
    w, v = eigsh(H, k=k, which='SM')
    for i in range(k):
        print(f'E{i} = {w[i]}')
        print(f'|Ψ{i}> = {v[:,i]}')