from functools import reduce

from hypothesis import given
import hypothesis.strategies as st
import numpy as np
import pytest

import qsim.gate
import qsim.state
from qsim.measurement import measure

from .common import (
    classical_bitstrings,
    n_qubits,
    rng,
    valid_states,
)

classical_states = classical_bitstrings.map(qsim.state.from_classical)
arbitrary_qubit = n_qubits.flatmap(lambda n: st.integers(0, n - 1))


@given(rng, arbitrary_qubit, classical_bitstrings)
def test_classical_measurement(rng, qubit, bitstring):
    """Test measurements on "classical" states.

    The outcome of a measurement on qubit 'n' of a classical state (i.e. a state
    constructed from a classical bitstring) is the value of the n-th bit of
    the bitstring.
    """
    classical_state = qsim.state.from_classical(bitstring)
    # [::-1] because qubits are indexed from the least-significant,
    # but bitstring are written most-significant bit first.
    expected_outcome = bool(int(bitstring[::-1][qubit]))
    outcome, _ = measure(rng, qubit, classical_state)
    assert outcome == expected_outcome


@given(rng, arbitrary_qubit, valid_states)
def test_subsequent_measurement(rng, qubit, ket):
    """Test that subsequent measurements of the same qubit yield the same outcome."""
    outcome, measured_ket = measure(rng, qubit, ket)
    outcome2, measured_ket2 = measure(rng, qubit, measured_ket)
    assert np.allclose(measured_ket, measured_ket2)
    assert outcome == outcome2
