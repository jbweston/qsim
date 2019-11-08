import string
from functools import partial

from hypothesis import given
import hypothesis.strategies as st
import numpy as np
import pytest

import qsim.state


MAX_QUBITS = 5

n_qubits = st.integers(1, MAX_QUBITS)
classical_bitstrings = n_qubits.flatmap(
    lambda n: st.integers(0, 2 ** n).map(partial("{:0{n}b}".format, n=n))
)
invalid_bitstrings = n_qubits.flatmap(
    lambda n: st.text(alphabet=string.digits, min_size=n, max_size=n)
).filter(lambda s: any(b not in "01" for b in s))


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
