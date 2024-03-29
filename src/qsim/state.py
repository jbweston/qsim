"""Quantum state vectors

The quantum state of :math:`n` quantum bits is represented as a 1D array of complex
numbers of length :math:`2^n`; the components of the state vector in the
computational basis.

The computational basis for :math:`n` qubits is ordered by the number represented
by the associated classical bitstring.
"""

import numpy as np

__all__ = ["from_classical", "is_normalized", "is_normalizable", "normalize", "num_qubits", "zero"]  # type: ignore


def from_classical(bitstring):
    """Return a quantum state corresponding to a classical bitstring.

    Parameters
    ----------
    bitstring : sequence of bits
        Can be a string like "01011", or a sequence of
        integers.

    Returns
    -------
    state : ndarray[(2**n,), complex]
        The state vector in the computational basis.
        Has :math:`2^n` components.
    """
    bitstring = "".join(map(str, bitstring))
    n_qubits = len(bitstring)
    try:
        index = int(bitstring, base=2)
    except ValueError:
        raise ValueError("Input is not a classical bitstring") from None

    state = np.zeros(1 << n_qubits, dtype=complex)
    state[index] = 1
    return state


def zero(n: int):
    """Return the zero state on 'n' qubits."""
    state = np.zeros(1 << n, dtype=complex)
    state[0] = 1
    return state


def num_qubits(state):
    """Return the number of qubits in the state.

    Raises ValueError if 'state' does not have a shape that is
    an integer power of 2.
    """
    _check_valid_state(state)
    return state.shape[0].bit_length() - 1


def is_normalized(state: np.ndarray) -> bool:
    """Return True if and only if 'state' is normalized."""
    return np.isclose(np.linalg.norm(state), 1)


def is_normalizable(v: np.ndarray) -> bool:
    """Return True if and only if 'v' is normalizable."""
    # If the norm is too small then normalizing a vector using
    # the norm will yield a vector that is not normalized to machine
    # precision.
    return bool(np.linalg.norm(v) ** 2 > np.finfo(float).smallest_normal)


def normalize(state: np.ndarray) -> np.ndarray:
    """Return a normalized state, given a potentially un-normalized one."""
    if not is_normalizable(state):
        raise ValueError("State is not normalizable")
    return state / np.linalg.norm(state)


def _check_valid_state(state):
    if not (
        # is an array
        isinstance(state, np.ndarray)
        # is complex
        and np.issubdtype(state.dtype, np.complex128)
        # is a vector
        and len(state.shape) == 1
        # has size 2**n, n > 1
        and np.log2(state.shape[0]).is_integer()
        and state.shape[0].bit_length() > 1
        and is_normalized(state)
    ):
        raise ValueError("State is not valid")
