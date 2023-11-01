from functools import partial
import string

import hypothesis.strategies as st
from hypothesis.extra import numpy as hnp
import numpy as np

# Numbers


def is_unit(c: complex) -> bool:
    return np.isclose(abs(c), 1)


MAX_QUBITS = 6

phases = st.floats(
    min_value=0, max_value=2 * np.pi, allow_nan=False, allow_infinity=False
)
valid_complex = st.complex_numbers(
    max_magnitude=1e10, allow_infinity=False, allow_nan=False
)
unit_complex = phases.map(lambda p: np.exp(1j * p))
nonunit_complex = valid_complex.filter(lambda c: not is_unit(c))

n_qubits = st.shared(st.integers(min_value=1, max_value=MAX_QUBITS))

rng = st.integers(0, 2**32 - 1).map(lambda n: np.random.default_rng(n))


# Gates


# Choose which qubits from 'n_qubits' to operate on with a gate that
# operates on 'gate_size' qubits
def select_n_qubits(gate_size: int):
    """Return a function that, given 'n_qubits', select 'gate_size' qubits from 0-'n_qubits'."""

    def _strat(n_qubits):
        assert n_qubits >= gate_size
        possible_qubits = st.integers(0, n_qubits - 1)
        return st.lists(
            possible_qubits, min_size=gate_size, max_size=gate_size, unique=True
        ).map(tuple)

    return _strat


def unitary(n_qubits):
    """Return a strategy for generating unitary matrices of size 2**n_qubits."""
    size = 1 << n_qubits
    return (
        hnp.arrays(complex, (size, size), elements=valid_complex)
        .map(lambda a: np.linalg.qr(a)[0])
        .filter(lambda u: np.all(np.isfinite(u)))
    )


single_qubit_gates = unitary(1)
two_qubit_gates = unitary(2)
n_qubit_gates = n_qubits.flatmap(unitary)


# States


def ket(n_qubits):
    return normalized_array(1 << n_qubits)


def is_normalizable_to_full_precision(v):
    # If the squared norm is > 0 but sub-normal
    # (https://en.wikipedia.org/wiki/Subnormal_number)
    # then normalizing the vector with v/|v| will not yield a vector
    # that is normalized to machine precision.
    return np.linalg.norm(v) ** 2 > np.finfo(float).smallest_normal


def normalized_array(size):
    return (
        hnp.arrays(complex, (size,), elements=valid_complex)
        .filter(is_normalizable_to_full_precision)  # vectors must be normalizable
        .map(lambda v: v / np.linalg.norm(v))
    )


state_dimensions = n_qubits.map(lambda n: 2**n)
state_shapes = state_dimensions.map(lambda x: (x,))

valid_states = n_qubits.flatmap(ket)
zero_states = state_shapes.map(lambda s: np.zeros(s, complex))

invalid_state_dimensions = st.integers(0, 2**MAX_QUBITS).filter(
    lambda n: not np.log2(n).is_integer()
)
invalid_shape_states = invalid_state_dimensions.flatmap(normalized_array)
invalid_norm_states = valid_states.flatmap(
    lambda x: nonunit_complex.map(lambda c: c * x)
)

invalid_states = st.one_of(invalid_shape_states, invalid_norm_states, zero_states)


# Bitstrings

classical_bitstrings = n_qubits.flatmap(
    lambda n: st.integers(0, 2**n).map(partial("{:0{n}b}".format, n=n))
)
invalid_bitstrings = n_qubits.flatmap(
    lambda n: st.text(alphabet=string.digits, min_size=n, max_size=n)
).filter(lambda s: any(b not in "01" for b in s))
