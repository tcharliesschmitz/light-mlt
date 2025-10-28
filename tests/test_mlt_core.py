import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


import numpy as np
from light_mlt.light_mlt import (
    to_base_p_matrix, from_base_p_matrix,
    gauss_jordan_inv_mod_p, random_invertible_matrix_mod_p
)

def test_base_p_roundtrip():
    ids = np.array([1, 5, 12])
    D = to_base_p_matrix(ids, p=5, n=3)
    ids_rec = from_base_p_matrix(D, p=5)
    assert np.all(ids == ids_rec)

def test_invertible_matrix_mod_p():
    p = 13
    rng = np.random.default_rng(42)
    M = random_invertible_matrix_mod_p(4, p, rng)
    Minv = gauss_jordan_inv_mod_p(M, p)
    I = (M @ Minv) % p
    assert np.all(I == np.eye(4, dtype=int))
