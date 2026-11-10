"""
Encoding: build the ansatz for the QAP

Unary Encoding: Bit i,j being 1 means facilitiy I goes to location J.
The resulting bitstring will be a permutation matrix.
One bit active in each row and column, two or more means its an
invalid solution. Uses N^2 bits.

"""
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
import numpy as np

def n_qubits(n_cities: int) -> int:
    """Number of qubits needed to represent an answer.

    Args:
        n_cities: Number of cities n.

    Returns:
        (n)^2.
    """
    return (n_cities) ** 2


def bits_from_key(key: str) -> str:
    """Convert a Qiskit counts key, with qubit 0 rightmost, to qubit order.

    Args:
        key: Bitstring from ``get_counts``.

    Returns:
        The bitstring with qubit 0 first.
    """
    return key.replace(" ", "")[::-1]


def decode_ans(bits: str, n_facilities: int) -> list[int] | None:
    """Answer from a bitstring.

    Qubit (c) * (n) + (t) holds x[c, t], so the table is stored
    row by row, one row per city.

    Args:
        bits: Bitstring in qubit order.
        n_facilities: Number of facilities n.

    Returns:
        The solution [0, c1, c2, ...], or None if the bits are not a valid solution.
    """
    m = n_facilities 
    x = [
        [int(bits[(c) * m + (t )]) for t in range(1, n_facilities)]
        for c in range(1, n_facilities)
    ]
    rows_ok = all(sum(row) == 1 for row in x)
    cols_ok = all(sum(col) == 1 for col in zip(*x))
    if not (rows_ok and cols_ok):
        return None
    order = [0] * m
    for c, row in enumerate(x, start=1):
        order[row.index(1)] = c
    return [0, *order]


def total_cost(tour: list[int], dist, flow) -> float:
    """Total cost of the solution

    Args:
        tour: Facilities in placment order.
        dist: Distance matrix, dist[a, b] = Distance between location A and location B.
        flow: Flow matrix, flow[a,b] = Flow between facilitiy A and facility B.

    Returns:
        The total cost.
    """
    return np.sum(dist * flow) / 2


def build_ansatz(cost: SparsePauliOp, reps: int = 2) -> QAOAAnsatz:
    """QAOA ansatz with one beta and one gamma per layer.

    Args:
        cost: Cost Hamiltonian H_C.
        reps: Number of QAOA layers p.

    Returns:
        The QAOA circuit with 2 * reps symbolic angles.
    """
    return QAOAAnsatz(cost, reps=reps)
