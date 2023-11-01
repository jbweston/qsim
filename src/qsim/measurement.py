"""Quantum measurements"""

from typing import Tuple

import numpy as np

from . import operator
from .state import normalize, num_qubits

__all__ = ["measure"]  # type: ignore


def measure(
    rng: np.random.RandomState, qubit: int, state: np.ndarray
) -> Tuple[bool, np.ndarray]:
    """Measure the given qubit in the computational basis."""
    nqs = num_qubits(state)
    if qubit >= nqs:
        raise ValueError("Cannot measure qubit {qubit} of an {nqs}-qubit state.")

    proj_1 = operator.apply(m_1, [qubit], state)

    p_1 = state.conj() @ proj_1

    if rng.random() < p_1:
        return (True, normalize(proj_1))
    else:
        proj_0 = operator.apply(m_0, [qubit], state)
        p_0 = 1 - p_1
        return (False, normalize(proj_0))


m_1 = np.array([[0, 0], [0, 1]], complex)
m_0 = np.array([[1, 0], [0, 0]], complex)
