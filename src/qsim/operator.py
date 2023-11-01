import numpy as np

from .state import num_qubits

__all__ = ["apply", "is_hermitian", "is_unitary", "is_valid", "n_qubits"]


def apply(op, qubits, state):
    """Apply an operator to the specified qubits of a state

    Parameters
    ----------
    op: ndarray[complex]
        The operator to apply.
    qubits : sequence of int
        The qubits on which to act. Qubit 0 is the least significant qubit.
    state : ndarray[complex]

    Returns
    -------
    new_state : ndarray[complex]
    """
    _check_apply(op, qubits, state)

    n_op_qubits = n_qubits(op)
    n_state_qubits = num_qubits(state)

    # We can view an n-qubit op as a 2*n-tensor (n contravariant and n contravariant
    # indices) and an n-qubit state as an n-tensor (contravariant indices)
    # with each axis having length 2 (the state space of a single qubit).
    op = op.reshape((2,) * 2 * n_op_qubits)
    state = state.reshape((2,) * n_state_qubits)

    # Our qubits are labeled from least significant to most significant, i.e. our
    # computational basis (for e.g. 2 qubits) is ordered like |00⟩, |01⟩, |10⟩, |11⟩.
    # We represent the state as a tensor in *row-major* order; this means that the
    # axis ordering is *backwards* compared to the qubit ordering (the least significant
    # qubit corresponds to the *last* axis in the tensor etc.)
    qubit_axes = tuple(n_state_qubits - 1 - np.asarray(qubits))

    # Applying the op to the state vector is then the tensor product over the
    # appropriate axes.
    axes = (np.arange(n_op_qubits, 2 * n_op_qubits), qubit_axes)
    new_state = np.tensordot(op, state, axes=axes)

    # tensordot effectively re-orders the qubits such that the qubits we operated
    # on are in the most-significant positions (i.e. their axes come first). We
    # thus need to transpose the axes to place them back into their original positions.
    untouched_axes = tuple(
        idx for idx in range(n_state_qubits) if idx not in qubit_axes
    )
    inverse_permutation = np.argsort(qubit_axes + untouched_axes)
    return np.transpose(new_state, inverse_permutation).reshape(-1)


def _all_distinct(elements):
    if not elements:
        return True
    elements = iter(elements)
    fst = next(elements)
    return all(fst != x for x in elements)


def _check_apply(op, qubits, state):
    if not _all_distinct(qubits):
        raise ValueError("Cannot apply an operator to repeated qubits.")

    n_op_qubits = n_qubits(op)
    if n_op_qubits != len(qubits):
        raise ValueError(
            f"Cannot apply an {n_op_qubits}-qubit operator to {len(qubits)} qubits."
        )

    n_state_qubits = num_qubits(state)

    if n_op_qubits > n_state_qubits:
        raise ValueError(
            f"Cannot apply an {n_op_qubits}-qubit operator "
            f"to an {n_state_qubits}-qubit state."
        )

    invalid_qubits = [q for q in qubits if q >= n_state_qubits]
    if invalid_qubits:
        raise ValueError(
            f"Cannot apply operator to qubits {invalid_qubits} "
            f"of an {n_state_qubits}-qubit state."
        )


def is_hermitian(op: np.ndarray) -> bool:
    """Return True if and only if 'op' is a valid Hermitian operator."""
    return is_valid(op) and np.allclose(op, op.conj().T)


def is_unitary(op: np.ndarray) -> bool:
    """Return True if and only if 'op' is a valid unitary operator."""
    return is_valid(op) and np.allclose(op @ op.conj().T, np.identity(op.shape[0]))


def is_valid(op: np.ndarray) -> bool:
    """Return True if and only if 'op' is a valid operator."""
    return (
        # is an array
        isinstance(op, np.ndarray)
        # is complex
        and np.issubdtype(op.dtype, np.complex128)
        # is square
        and op.shape[0] == op.shape[1]
        # has size 2**n, n > 1
        and np.log2(op.shape[0]).is_integer()
        and op.shape[0].bit_length() > 1
    )


def _check_valid_operator(op):
    if not is_valid(op):
        raise ValueError("Operator is invalid")


def n_qubits(op):
    """Return the number of qubits that the operator acts on.

    Raises ValueError if 'o' does not have a shape that is
    an integer power of 2.
    """
    _check_valid_operator(op)
    return op.shape[0].bit_length() - 1
