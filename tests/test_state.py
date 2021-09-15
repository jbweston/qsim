import string
from functools import partial

from hypothesis import given
from hypothesis.extra import numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest

import qsim.state


MAX_QUBITS = 5


def is_unit(c: complex):
    return np.isclose(abs(c), 1)


n_qubits = st.integers(1, MAX_QUBITS)
state_dimensions = n_qubits.map(lambda n: 2 ** n)
state_shapes = state_dimensions.map(lambda x: (x,))
classical_bitstrings = n_qubits.flatmap(
    lambda n: st.integers(0, 2 ** n).map(partial("{:0{n}b}".format, n=n))
)
invalid_bitstrings = n_qubits.flatmap(
    lambda n: st.text(alphabet=string.digits, min_size=n, max_size=n)
).filter(lambda s: any(b not in "01" for b in s))

invalid_state_dimensions = st.integers(0, 2 ** MAX_QUBITS).filter(
    lambda n: not np.log2(n).is_integer()
)


valid_complex = st.complex_numbers(
    max_magnitude=1e10, allow_infinity=False, allow_nan=False
)
nonunit_complex = valid_complex.filter(lambda c: not is_unit(c))


def ket(n_qubits):
    return normalized_array(1 << n_qubits)


def normalized_array(size):
    return (
        hnp.arrays(complex, (size,), elements=valid_complex)
        .filter(lambda v: np.linalg.norm(v) > 0)  # vectors must be normalizable
        .map(lambda v: v / np.linalg.norm(v))
    )


valid_states = n_qubits.flatmap(ket)
invalid_shape_states = invalid_state_dimensions.flatmap(normalized_array)
invalid_norm_states = valid_states.flatmap(
    lambda x: nonunit_complex.map(lambda c: c * x)
)
zero_states = state_shapes.map(lambda s: np.zeros(s, complex))
invalid_states = st.one_of(invalid_shape_states, invalid_norm_states, zero_states)


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
