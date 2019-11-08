"""Quantum state vectors

The quantum state of :math:`n` quantum bits is represented as a 1D array of complex
numbers of length :math:`2^n`; the components of the state vector in the
computational basis.

The computational basis for :math:`n` qubits is ordered by the number represented
by the associated classical bitstring.
"""

import numpy as np

__all__ = ["from_classical"]


def from_classical(bitstring):
    """Return a quantum state corresponding to a classical bitstring.

    Parameters
    ----------
    bitstring : sequence of bits
        Can be a string like "01011", or a sequence of
        integers.

    Returns
    -------
    state : ndarray[complex]
        The state vector in the computational basis.
        Has :math:`2^n` components.
    """

    bitstring = "".join(map(str, bitstring))
    n = len(bitstring)
    try:
        index = int(bitstring, base=2)
    except ValueError:
        raise ValueError("Input is not a classical bitstring") from None

    state = np.zeros(2 ** n, dtype=complex)
    state[index] = 1
    return state
