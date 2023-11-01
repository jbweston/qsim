from hypothesis import given
import numpy as np
import pytest

import qsim.state

from .common import (
    classical_bitstrings,
    invalid_bitstrings,
    invalid_states,
    invalid_norm_states,
    n_qubits,
    valid_states,
)


@given(classical_bitstrings)
def test_from_classical(bitstring):
    state = qsim.state.from_classical(bitstring)

    i = int(bitstring, base=2)

    assert np.issubdtype(state.dtype, np.dtype(complex))
    assert state.shape == (2 ** len(bitstring),)
    assert np.linalg.norm(state) == 1
    assert abs(state[i]) == 1


@given(classical_bitstrings)
def test_from_classical_works_on_integer_lists(bitstring):
    int_list = [int(b) for b in bitstring]

    assert np.all(
        qsim.state.from_classical(bitstring) == qsim.state.from_classical(int_list)
    )


@given(invalid_bitstrings)
def test_from_classical_raises_on_bad_input(bitstring):
    with pytest.raises(ValueError):
        qsim.state.from_classical(bitstring)


@given(n_qubits)
def test_zero(n):
    assert np.array_equal(qsim.state.zero(n), qsim.state.from_classical("0" * n))


@given(classical_bitstrings)
def test_num_qubits(s):
    state = qsim.state.from_classical(s)
    assert qsim.state.num_qubits(state) == len(s)


@given(invalid_states)
def test_num_qubits_raises_exception(state):
    with pytest.raises(ValueError):
        qsim.state.num_qubits(state)


@given(invalid_norm_states)
def test_is_not_normalized(state):
    assert not qsim.state.is_normalized(state)


@given(valid_states)
def test_is_normalized(state):
    assert qsim.state.is_normalized(state)


@given(n_qubits)
def test_zero(n):
    z = qsim.state.zero(n)
    assert qsim.state.num_qubits(z) == n
