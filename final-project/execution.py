"""Execution: build the QAOA loop and run the QAP.

The variational loop:

    1. A classical optimizer (COBYLA) proposes the QAOA angles (beta, gamma).
    2. The Estimator runs the circuit and returns the average energy <H>.
    3. The optimizer lowers <H> and goes back to step 1.

Because the landscape has many local minima, the loop is restarted from a
few random points. The best angles found are then used to sample the circuit,
and the counts go to analysis.py.

Called from main.py. With no backend name it runs on the Aer simulator;
with one it runs on IBM hardware inside a single Session.
"""

import time

import numpy as np
from encoding import build_ansatz
from ops import build_hamiltonian
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from scipy.optimize import minimize


def get_primitives(backend_name: str | None):
    """Create the backend, the Estimator, the Sampler and, on hardware, a Session.

    Args:
        backend_name: IBM backend name, or None for the Aer simulator.

    Returns:
        (backend, estimator, sampler, session). The session is None on the
        simulator.
    """
    if backend_name is None:
        return AerSimulator(), AerEstimator(), AerSampler(), None
    # Only hardware runs need qiskit_ibm_runtime and a saved IBM account.
    from qiskit_ibm_runtime import (
        EstimatorV2,
        QiskitRuntimeService,
        SamplerV2,
        Session,
    )

    backend = QiskitRuntimeService().backend(backend_name)
    # A Session keeps our place at the front of the queue for the whole loop.
    # Without it, every step (which needs the previous result) would wait in
    # line again.
    session = Session(backend=backend)
    return backend, EstimatorV2(mode=session), SamplerV2(mode=session), session


def run_qaoa(estimator, circuit, hamiltonian, x0, shots: int, maxiter: int):
    """Minimize <H>(theta) with COBYLA.

    Args:
        estimator: Estimator primitive that returns <H>.
        circuit: Transpiled QAOA circuit.
        hamiltonian: Cost Hamiltonian on the same qubit layout as ``circuit``.
        x0: Starting angles (beta, gamma) in radians.
        shots: Shots per energy estimate.
        maxiter: Maximum optimizer iterations.

    Returns:
        (optimizer result, list of every energy evaluated).
    """
    history = []
    # The Estimator takes a target standard error rather than a shot count.
    # For an observable with values near +-1, N shots give about 1 / sqrt(N).
    precision = 1 / np.sqrt(shots)

    def cost(theta):
        """Energy at the angles ``theta``, recorded in ``history``."""
        job = estimator.run([(circuit, hamiltonian, theta, precision)])
        history.append(float(job.result()[0].data.evs))
        return history[-1]

    result = minimize(cost, x0, method="COBYLA", options={"maxiter": maxiter})
    return result, history


def run(
    names,
    dist,
    backend_name=None,
    reps=2,
    starts=3,
    shots=4096,
    maxiter=150,
    seed=1,
) -> dict:
    """Optimize the QAOA circuit, sample it, and return the results.

    Args:
        names: City names.
        dist: Cost matrix, dist[a, b] = cost of going from city a to city b.
        backend_name: IBM backend name, or None for the Aer simulator.
        reps: Number of QAOA layers p.
        starts: Random restarts of the optimizer.
        shots: Shots per energy estimate and for the final sampling.
        maxiter: Maximum optimizer iterations per restart.
        seed: Seed for the random starting angles.

    Returns:
        The problem, the settings, the energy history of every restart, the
        best angles, and the sampled counts, as ``analysis.analyze`` expects.
    """
    hamiltonian = build_hamiltonian(dist)
    ansatz = build_ansatz(hamiltonian, reps)
    backend, estimator, sampler, session = get_primitives(backend_name)

    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    # Transpile once: pick physical qubits and rewrite the gates into the
    # backend's native set. The angles stay symbolic, so this serves every
    # step of the loop. The Hamiltonian must follow the same qubit layout.
    isa = pm.run(ansatz)
    isa_h = hamiltonian.apply_layout(isa.layout)
    print(
        f"{isa.num_qubits} qubits, depth {isa.depth()}, "
        f"{isa.num_nonlocal_gates()} two-qubit gates on {backend.name}"
    )

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best, history = None, []
    try:
        for s in range(starts):
            # QAOAAnsatz orders its parameters beta_0..beta_(p-1), then
            # gamma_0..gamma_(p-1). Small gamma and moderate beta is a
            # common starting point.
            x0 = np.concatenate(
                [rng.uniform(0, np.pi / 2, reps), rng.uniform(0, 0.5, reps)]
            )
            result, hist = run_qaoa(estimator, isa, isa_h, x0, shots, maxiter)
            print(
                f"  start {s}: best <H> = {min(hist):+.3f} "
                f"({len(hist)} evaluations)"
            )
            history.append(hist)
            if best is None or result.fun < best.fun:
                best = result
        # Fix the best angles, add measurements, and sample the circuit.
        measured = ansatz.assign_parameters(best.x)
        measured.measure_all()
        counts = (
            sampler.run([pm.run(measured)], shots=shots)
            .result()[0]
            .data.meas.get_counts()
        )
    finally:
        if session is not None:
            session.close()

    return {
        "cities": list(names),
        "costs": np.asarray(dist).tolist(),
        "backend": backend.name,
        "reps": reps,
        "shots": shots,
        "seconds": time.perf_counter() - t0,
        "theta": best.x.tolist(),
        "history": history,
        "counts": counts,
    }
