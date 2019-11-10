"""Quantum gate operations

A quantum gate acting on :math:`n` qubits is a :math:`2^n×2^n` unitary
matrix written in the computational basis.
"""

import numpy as np

__all__ = [
    "n_qubits",
    "controlled",
    # -- Single qubit gates --
    "id",
    "x",
    "y",
    "z",
    "not_",
    "sqrt_not",
    "phase_shift",
    # -- 2 qubit gates --
    "cnot",
    "swap",
    # -- 3 qubit gates --
    "toffoli",
    "cswap",
    "fredkin",
    "deutsch",
]  # type: ignore


def _check_valid_gate(gate):
    if not (
        # is an array
        isinstance(gate, np.ndarray)
        # is complex
        and np.issubdtype(gate.dtype, np.complex128)
        # is square
        and gate.shape[0] == gate.shape[1]
        # has size 2**n, n > 1
        and np.log2(gate.shape[0]).is_integer()
        and np.log2(gate.shape[0]) > 0
        # is unitary
        and np.allclose(gate @ gate.conjugate().transpose(), np.identity(gate.shape[0]))
    ):
        raise ValueError("Gate is not valid")


def n_qubits(gate):
    """Return the number of qubits that a gate acts on.

    Raises ValueError if 'gate' does not have a shape that is
    an integer power of 2.
    """
    _check_valid_gate(gate)
    n = np.log2(gate.shape[0])
    assert n.is_integer()
    return int(n)


def controlled(gate):
    """Return a controlled quantum gate, given a quantum gate.

    If 'gate' operates on :math:`n` qubits, then the controlled gate operates
    on :math:`n+1` qubits, where the most-significant qubit is the control.

    Parameters
    ----------
    gate : np.ndarray[complex]
        A quantum gate acting on :math:`n` qubits;
        a :math:`2^n×2^n` unitary matrix in the computational basis.

    Returns
    -------
    controlled_gate : np.ndarray[(2**(n+1), 2**(n+1)), complex]
    """
    _check_valid_gate(gate)
    n = gate.shape[0]
    zeros = np.zeros((n, n))
    return np.block([[np.identity(n), zeros], [zeros, gate]])


# -- Single qubit gates --

#: The identity gate on 1 qubit
id = np.identity(2, complex)
#: Pauli X gate
x = np.array([[0, 1], [1, 0]], complex)
#: NOT gate
not_ = x
#: Pauli Y gate
y = np.array([[0, -1j], [1j, 0]], complex)
#: Pauli Z gate
z = np.array([[1, 0], [0, -1]], complex)
#: SQRT(NOT) gate
sqrt_not = 0.5 * (1 + 1j * id - 1j * x)
#: Hadamard gate
hadamard = np.sqrt(0.5) * (x + z)


def phase_shift(phi):
    "Return a gate that shifts the phase of :math:`|1⟩` by :math:`φ`."
    return np.array([[1, 0], [0, np.exp(1j * phi)]])


# -- Two qubit gates --

#: Controlled NOT gate
cnot = controlled(x)
#: SWAP gate
swap = np.identity(4, complex)[:, (0, 2, 1, 3)]

# -- Three qubit gates --

#: Toffoli (CCNOT) gate
toffoli = controlled(cnot)
#: Controlled SWAP gate
cswap = controlled(swap)
#: Fredkin gate
fredkin = cswap


def deutsch(phi):
    "Return a Deutsch gate for angle :math:`φ`."
    gate = np.identity(8, complex)
    gate[-2:, -2:] = np.array(
        [[1j * np.cos(phi), np.sin(phi)], [np.sin(phi), 1j * np.cos(phi)]]
    )
    return gate
