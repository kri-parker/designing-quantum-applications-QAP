"""Ops: build the Hamiltonian for the traveling salesman example.

The problem: visit every city once and return to the start, as cheaply as
possible. We write it as a QUBO (quadratic unconstrained binary optimization)

    minimize  E(x) = x^T Q x + offset,   x_i in {0, 1}

and convert it to an Ising Hamiltonian with x_i = (1 - Z_i) / 2, so the
ground state of H is the cheapest tour.

Binary variables (see encoding.py): x[c, t] = 1 means city c is at position
t. City 0 is fixed at position 0. A tour is a loop with no natural starting
point, so fixing one city removes rotated copies of the same tour and saves
qubits. Four cities leave 3 x 3 = 9 variables.

The input is a table of travel costs, d[a, b] = cost of going from city a to
city b. The table does not have to be symmetric.
"""

import json
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def load_problem(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Read a problem file of the form {"cities": [names], "costs": [[...]]}.

    Args:
        path: Path of the JSON file.

    Returns:
        (names, costs): the city names and the cost matrix.

    Raises:
        ValueError: If the cost matrix is not square with one row per city,
            or there are fewer than 3 cities.
    """
    data = json.loads(Path(path).read_text())
    names, costs = data["cities"], np.array(data["costs"], dtype=float)
    if costs.shape != (len(names), len(names)):
        raise ValueError(
            f"costs must be {len(names)}x{len(names)}, got {costs.shape}"
        )
    if len(names) < 3:
        raise ValueError("need at least 3 cities")
    return names, costs


def var(city: int, pos: int, n: int) -> int:
    """Qubit index of the bit x[city, pos].

    Args:
        city: City, 1 to n - 1.
        pos: Step of the tour, 1 to n - 1.
        n: Number of cities.

    Returns:
        The qubit index.
    """
    return (city - 1) * (n - 1) + (pos - 1)


def tsp_qubo(dist: np.ndarray, penalty: float) -> tuple[np.ndarray, float]:
    """QUBO matrix of the traveling salesman problem.

    Travel cost: add d(a, b) when city a is at step t and city b is at step
    t + 1, plus the legs out of and back to the fixed city 0.

    Penalty: for each city, A * (1 - sum_t x[c,t])^2 is zero only if the city
    is visited exactly once. For each step, A * (1 - sum_c x[c,t])^2 is zero
    only if exactly one city is there.

    Args:
        dist: Cost matrix, dist[a, b] = cost of going from city a to city b.
        penalty: Penalty weight A.

    Returns:
        (Q, offset) with Q upper triangular, so E(x) = x^T Q x + offset.
    """
    n = len(dist)
    m = n - 1
    Q = np.zeros((m * m, m * m))
    offset = 0.0
    cities = range(1, n)

    def add(i, j, w):
        """Add w * x_i * x_j to Q.

        For i == j this is w * x_i, because a bit squared is itself, so linear
        terms live on the diagonal of Q.
        """
        Q[min(i, j), max(i, j)] += w

    for c in cities:
        add(var(c, 1, n), var(c, 1, n), dist[0, c])  # leave city 0
        add(var(c, m, n), var(c, m, n), dist[c, 0])  # return to city 0
        for c2 in cities:
            if c2 != c:
                for t in range(1, m):
                    add(var(c, t, n), var(c2, t + 1, n), dist[c, c2])

    # Each group of bits must hold exactly one 1. Expanding the square gives
    # (1 - sum x)^2 = 1 - sum x_i + 2 sum_{i<j} x_i x_j.
    one_city_all_steps = [[var(c, t, n) for t in cities] for c in cities]
    one_step_all_cities = [[var(c, t, n) for c in cities] for t in cities]
    for g in one_city_all_steps + one_step_all_cities:
        offset += penalty
        for a, i in enumerate(g):
            add(i, i, -penalty)
            for j in g[a + 1 :]:
                add(i, j, 2 * penalty)
    return Q, offset


def qubo_to_ising(Q: np.ndarray, offset: float) -> SparsePauliOp:
    """Substitute x = (1 - Z) / 2 to turn a QUBO into an Ising Hamiltonian.

        x_i       -> (1 - Z_i) / 2
        x_i x_j   -> (1 - Z_i - Z_j + Z_i Z_j) / 4

    Args:
        Q: Upper-triangular QUBO matrix.
        offset: Constant term of the QUBO.

    Returns:
        sum h_i Z_i + sum J_ij Z_i Z_j + const as a sum of Pauli strings.
    """
    n = len(Q)
    h, const, terms = np.zeros(n), offset, []
    for i in range(n):
        h[i] -= Q[i, i] / 2
        const += Q[i, i] / 2
        for j in range(i + 1, n):
            if Q[i, j]:
                h[i] -= Q[i, j] / 4
                h[j] -= Q[i, j] / 4
                const += Q[i, j] / 4
                terms.append((_z_label(n, i, j), Q[i, j] / 4))
    terms += [(_z_label(n, i), h[i]) for i in range(n) if h[i]]
    terms.append(("I" * n, const))
    return SparsePauliOp.from_list(terms).simplify()


def _z_label(n: int, *qubits: int) -> str:
    """Pauli label with Z on the given qubits and I elsewhere.

    Args:
        n: Total number of qubits.
        *qubits: Qubits that get a Z.

    Returns:
        The label in Qiskit order, qubit 0 on the right.
    """
    label = ["I"] * n
    for q in qubits:
        label[q] = "Z"
    # Qiskit labels put qubit 0 on the right, so reverse.
    return "".join(reversed(label))


def default_penalty(dist: np.ndarray) -> float:
    """Penalty weight large enough that breaking a rule never pays.

    Args:
        dist: Cost matrix.

    Returns:
        Twice the longest leg.
    """
    return 2 * float(dist.max())


def build_hamiltonian(
    dist: np.ndarray, penalty: float | None = None
) -> SparsePauliOp:
    """Cost Hamiltonian for a cost matrix.

    Costs are divided by the longest leg first. That only rescales the
    energies and leaves the ranking of tours unchanged, but it keeps the QAOA
    angles in a sensible range whatever units the costs use.

    Args:
        dist: Cost matrix.
        penalty: Penalty weight for the scaled costs, or None for
            ``default_penalty``.

    Returns:
        The Ising Hamiltonian H_C.
    """
    dist = np.asarray(dist) / np.max(dist)
    penalty = penalty if penalty is not None else default_penalty(dist)
    return qubo_to_ising(*tsp_qubo(dist, penalty))


def qubo_energy(bits: str, dist: np.ndarray, penalty: float) -> float:
    """Classical QUBO energy of a bitstring.

    It equals the diagonal entry of H for the same bitstring when ``dist`` is
    scaled the same way, which makes it a check on the Hamiltonian.

    Args:
        bits: Bitstring in qubit order.
        dist: Cost matrix.
        penalty: Penalty weight.

    Returns:
        The energy E(x).
    """
    x = np.array([int(b) for b in bits], dtype=float)
    Q, offset = tsp_qubo(dist, penalty)
    return float(x @ Q @ x + offset)
