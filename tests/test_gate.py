from functools import reduce

from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
import numpy as np
import pytest

import qsim.gate


# -- Strategies for generating values --


n_qubits = st.shared(st.integers(min_value=1, max_value=6))


# Choose which qubits from 'n_qubits' to operate on with a gate that
# operates on 'gate_size' qubits
def select_n_qubits(gate_size):
    def _strat(n_qubits):
        assert n_qubits >= gate_size
        possible_qubits = st.integers(0, n_qubits - 1)
        return st.lists(possible_qubits, gate_size, gate_size, unique=True).map(tuple)

    return _strat


valid_complex = st.complex_numbers(allow_infinity=False, allow_nan=False)
phases = st.floats(
    min_value=0, max_value=2 * np.pi, allow_nan=False, allow_infinity=False
)


def unitary(n_qubits):
    size = 1 << n_qubits
    return (
        hnp.arrays(complex, (size, size), valid_complex)
        .map(lambda a: np.linalg.qr(a)[0])
        .filter(lambda u: np.all(np.isfinite(u)))
    )


def ket(n_qubits):
    size = 1 << n_qubits
    return (
        hnp.arrays(complex, (size,), valid_complex)
        .filter(lambda v: np.linalg.norm(v) > 0)  # vectors must be normalizable
        .map(lambda v: v / np.linalg.norm(v))
    )


single_qubit_gates = unitary(1)
two_qubit_gates = unitary(2)
n_qubit_gates = n_qubits.flatmap(unitary)

# Projectors on the single qubit computational basis
project_zero = np.array([[1, 0], [0, 0]])
project_one = np.array([[0, 0], [0, 1]])


def product_gate(single_qubit_gates):
    # We reverse so that 'single_qubit_gates' can be indexed by the qubit
    # identifier; e.g. qubit #0 is actually the least-significant qubit
    return reduce(np.kron, reversed(single_qubit_gates))


# -- Tests --


@given(n_qubits, n_qubit_gates)
def test_n_qubits(n, gate):
    assert qsim.gate.n_qubits(gate) == n


@given(n_qubit_gates)
def test_n_qubits_invalid(gate):
    # Not a numpy array
    with pytest.raises(ValueError):
        qsim.gate.n_qubits(list(map(list, gate)))
    # Not complex
    with pytest.raises(ValueError):
        qsim.gate.n_qubits(gate.real)
    # Not square
    with pytest.raises(ValueError):
        qsim.gate.n_qubits(gate[:-2])
    # Not size 2**n, n > 0
    with pytest.raises(ValueError):
        qsim.gate.n_qubits(gate[:-1, :-1])
    # Not unitary
    nonunitary_part = np.zeros_like(gate)
    nonunitary_part[0, -1] = 1j
    with pytest.raises(ValueError):
        qsim.gate.n_qubits(gate + nonunitary_part)


@given(n_qubits, n_qubit_gates)
def test_controlled(n, gate):
    nq = 1 << n
    controlled_gate = qsim.gate.controlled(gate)
    assert controlled_gate.shape[0] == 2 * nq
    assert np.all(controlled_gate[:nq, :nq] == np.identity(nq))
    assert np.all(controlled_gate[nq:, nq:] == gate)


@given(phases)
def test_phase_gate_inverse(phi):
    assert np.allclose(
        qsim.gate.phase_shift(phi) @ qsim.gate.phase_shift(-phi), np.identity(2)
    )


@given(phases, st.integers())
def test_phase_gate_periodic(phi, n):
    atol = np.finfo(complex).resolution * abs(n)
    assert np.allclose(
        qsim.gate.phase_shift(phi),
        qsim.gate.phase_shift(phi + 2 * np.pi * n),
        atol=atol,
    )


@given(single_qubit_gates)
def test_id(gate):
    assert np.all(qsim.gate.id @ gate == gate)
    assert np.all(gate @ qsim.gate.id == gate)


def test_pauli_gates_are_involutary():
    pauli_gates = [qsim.gate.x, qsim.gate.y, qsim.gate.z]
    assert np.all(qsim.gate.x == qsim.gate.not_)
    for gate in pauli_gates:
        assert np.all(gate @ gate == qsim.gate.id)
    assert np.all(-1j * qsim.gate.x @ qsim.gate.y @ qsim.gate.z == qsim.gate.id)


def test_sqrt_not():
    assert np.all(qsim.gate.sqrt_not @ qsim.gate.sqrt_not == qsim.gate.not_)


def test_deutch():
    assert np.allclose(qsim.gate.deutsch(np.pi / 2), qsim.gate.toffoli)


def test_swap():
    assert np.all(qsim.gate.swap @ qsim.gate.swap == np.identity(4))


@given(single_qubit_gates, n_qubits.flatmap(ket), n_qubits.flatmap(select_n_qubits(1)))
def test_applying_single_gates(gate, state, selected):
    qubit, = selected
    n_qubits = state.shape[0].bit_length() - 1
    parts = [np.identity(2)] * n_qubits
    parts[qubit] = gate
    big_gate = product_gate(parts)

    should_be = big_gate @ state
    state = qsim.gate.apply(gate, [qubit], state)

    assert np.allclose(state, should_be)


@given(
    single_qubit_gates,
    n_qubits.filter(lambda n: n > 1).flatmap(ket),
    n_qubits.filter(lambda n: n > 1).flatmap(select_n_qubits(2)),
)
def test_applying_controlled_single_qubit_gates(gate, state, selected):
    control, qubit = selected
    n_qubits = state.shape[0].bit_length() - 1
    # When control qubit is |0⟩ the controlled gate acts like the identity on the other qubit
    parts_zero = [np.identity(2)] * n_qubits
    parts_zero[control] = project_zero
    parts_zero[qubit] = np.identity(2)
    # When control qubit is |1⟩ the controlled gate acts like the original gate on the other qubit
    parts_one = [np.identity(2)] * n_qubits
    parts_one[control] = project_one
    parts_one[qubit] = gate
    # The total controlled gate is then the sum of these 2 product gates
    big_gate = product_gate(parts_zero) + product_gate(parts_one)

    should_be = big_gate @ state
    state = qsim.gate.apply(qsim.gate.controlled(gate), [control, qubit], state)

    assert np.allclose(state, should_be)
