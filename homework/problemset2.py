import json
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZGate, YGate, XGate, HGate, CXGate
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler
from qiskit_ibm_runtime import Batch, Session
from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimator
from scipy.optimize import minimize
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from qiskit.quantum_info import Operator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00-Setup-Start-Here"))
from _job_state import (
    DEFAULT_POLL_INTERVAL,
    extract_counts,
    save_last_job,
    wait_for_job,
)
#problem 1
def prepQubitRotation(X : float, Y: float, Z: float) -> QuantumCircuit:
    qc = QuantumCircuit(1,1) # make circut
    qc.rx(X,[0]) #rotate in X
    qc.ry(Y,[0]) # rotate in Y
    qc.rz(Z,[0]) #rotate in Z
    return qc

#problem 1
def measureInGivenAxis(qc:QuantumCircuit,axis):
    if axis == 'X':
        qc.h(0)          # rotates the X axis onto Z
    if axis == 'Y':
        qc.sdg(0)        # rotates the Y axis onto X.
        qc.h(0)  
    #nothing for z

    #sv = Statevector(qc)  #prints statevecotr, cannot be run if qc.measure is called
    #sv.draw("latex")
    #print(sv)
    qc.measure(0,0)


DEFAULT_BACKEND = "ibm_rensselaer"

#copyed from provided files
def run_on_hardware(
    circuits: list[QuantumCircuit],
    shots: int,
    backend_name: str,
    account_name: str,
    preferred_instance: str = "MANE-4960-dedicated",
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> list[dict] | None:
    """Run a list of quantum circuits on IBM hardware via Qiskit Runtime.

    Transpiles every circuit for the target backend and submits them all as a
    single job (one PUB per circuit), then waits for results. The job ID is
    saved to disk for recovery if interrupted.

    Parameters
    ----------
    circuits : list[QuantumCircuit]
        The quantum circuits to run. All are submitted together in one job.
    shots : int
        Number of shots (repetitions) for sampling, applied to every circuit.
    backend_name : str
        Name of the IBM backend (e.g., "ibm_rensselaer").
    account_name : str
        Name of the saved IBM Quantum account.
    preferred_instance : str, optional
        Instance name to warn about if not active (default: PREFERRED_INSTANCE).
    poll_interval : int, optional
        Seconds between status checks (default: DEFAULT_POLL_INTERVAL).

    Returns
    -------
    list[dict] or None
        One dictionary per circuit (same order as ``circuits``) mapping
        bitstrings to counts if successful, None if the connection failed or
        the job was interrupted.
    """
    print(
        f"\n--- Running {len(circuits)} circuit(s) on IBM hardware backend "
        f"'{backend_name}', {shots} shots each ---"
    )
    start_time = time.perf_counter()

    try:
        service = QiskitRuntimeService(name=account_name)
        backend = service.backend(backend_name)
    except Exception as exc:  # noqa: BLE001 - report connection failure
        print(f"Could not reach backend '{backend_name}': {exc}")
        print(
            "Check that 02_save_token.py has been run, and run "
            "03_check_token.py to see which instances (and backends) you "
            "actually have access to. If the class backend has a different "
            f"name, pass it with: --backend <name>"
        )
        return None

    print(f"Connected to {backend.name} ({backend.num_qubits} qubits)")

    # Warn (don't block) if not on the preferred priority-queue instance.
    # Using a different instance just means longer wait times.
    # active_account()["instance"] is a CRN, not a friendly name, so look up
    # the matching name from the account's list of available instances.
    active_crn = service.active_account()["instance"]
    active_instance = next(
        (inst["name"] for inst in service.instances() if inst["crn"] == active_crn),
        active_crn,
    )
    if active_instance != preferred_instance:
        print(
            f"** Note **  You're using instance '{active_instance}', not the "
            f"preferred instance '{preferred_instance}'. Your job may "
            "wait longer in the queue than necessary.\n"
            "To switch instances:\n"
            "  1. Run: uv run 02_save_token.py\n"
            f"  2. When prompted, select '{preferred_instance}'\n"
            "  3. Re-run this script"
        )

    # Extra step, only needed for real hardware: convert to ISA
    # (Instruction Set Architecture) circuits that are transpiled into this
    # backend's actual basis gates and qubit connectivity. The simulator
    # above skipped this because it supports every gate directly and has
    # no connectivity restrictions. Passing a list transpiles them all at once.
    transpile_start = time.perf_counter()
    pass_manager = generate_preset_pass_manager(
        backend=backend, optimization_level=1
    )
    isa_circuits = pass_manager.run(circuits)
    transpile_time = time.perf_counter() - transpile_start
    print(f"Transpiled {len(isa_circuits)} circuit(s) in {transpile_time:.2f}s")

    # Steps 4-5 again, this time targeting real hardware: one PUB per
    # circuit, all in a single sampler.run() call so they share one job
    # and one queue wait.
    pubs = [(isa_circuit,) for isa_circuit in isa_circuits]
    sampler = RuntimeSampler(mode=backend)

    # Step 6: submit, then retrieve. Save the job ID to disk *before*
    # waiting. The queue can be long, and the job keeps running on IBM's
    # servers even if this script's connection drops while it waits.
    job = sampler.run(pubs, shots=shots)
    save_last_job(job.job_id(), backend.name, account_name)
    print(f"Submitted job {job.job_id()} ({len(pubs)} circuits) on {backend.name}.")
    print(
        "If this script is interrupted, retrieve the result later with:\n"
        "  uv run 05_retrieve_job.py"
    )
    print(f"Waiting for job results (checking every {poll_interval}s)...")

    try:
        result = wait_for_job(job, backend, poll_interval=poll_interval)
    except (KeyboardInterrupt, Exception) as exc:
        if isinstance(exc, KeyboardInterrupt):
            cause = "Cancelled."
        else:
            cause = f"Lost connection: {exc}"
        print(
            f"\n{cause} The job (id: {job.job_id()}) is still running on IBM's "
            "servers -- it doesn't need this script to stay connected.\n"
            "To pick it back up:\n"
            "  1. Reconnect (if needed), then open a terminal here.\n"
            "  2. Run:\n"
            "       uv run 05_retrieve_job.py\n"
            f"     (or:  uv run 05_retrieve_job.py {job.job_id()})\n"
            "  3. It will reconnect, show the job's status, and wait "
            "for it to finish if needed."
        )
        return None

    # Step 7: post-process, exactly as with the Aer result above, once per
    # PUB. result[i] lines up with circuits[i].
    all_counts = [extract_counts(pub_result)[1] for pub_result in result]

    elapsed = time.perf_counter() - start_time
    print(f"\n*** Hardware job completed! ***")
    for i, counts in enumerate(all_counts):
        print(f"Circuit {i} counts: {counts}")
    print(f"Total elapsed time: {elapsed:.2f}s")
    return all_counts

#problem 2
def convertObservableToPauli(M: np.ndarray) -> SparsePauliOp:
    """takes in a 2d matrix, converts to pauli"""
    op = Operator(M) #converts to quiskit operator
    return SparsePauliOp.from_operator(op) #converts to pauli expression 
#same thing but uses pauli strings
def convertStringToPauli(labels: str) -> SparsePauliOp:
    """takes in a pauli string (e.g. "XZ") or list of them, converts to pauli"""
    return SparsePauliOp(labels)

def main() -> int:
    qc1 = prepQubitRotation(np.pi/4,np.pi/3,np.pi/6)# prep qubits
    qc2 = prepQubitRotation(np.pi/4,np.pi/3,np.pi/6)# prep qubits
    qc3 = prepQubitRotation(np.pi/4,np.pi/3,np.pi/6)# prep qubits

    #problem 1
    
    measureInGivenAxis(qc1,'X') #measure in all 3 axis
    measureInGivenAxis(qc2,'Y')
    measureInGivenAxis(qc3,'Z') 

    counts = run_on_hardware([qc1,qc2,qc3],10000,DEFAULT_BACKEND,"default")# run on hardware

    plot_histogram(counts[0])#graph
    plot_histogram(counts[1])
    plot_histogram(counts[2])
    plt.show()
    
    """
    #problem 2
    #get obserables
    X_meas = np.array([[0, 1], [1, 0]])
    Y_meas = np.array([[0, -1j], [1j, 0]])
    Z_meas = np.array([[1, 0], [0, -1]])
    #covert to pauli
    X_pauli = convertObservableToPauli(X_meas)
    Y_pauli = convertObservableToPauli(Y_meas)
    Z_pauli = convertObservableToPauli(Z_meas)

    paulis = [X_pauli,Y_pauli,Z_pauli]
    #run estimators

    
    service = QiskitRuntimeService()
    backend = service.backend(DEFAULT_BACKEND)

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_qc = pm.run(qc1)
    isa_obs = [p.apply_layout(isa_qc.layout) for p in paulis]
    est = RuntimeEstimator(mode=backend)
    est.options.default_shots = 10000
    result = est.run([(isa_qc, isa_obs)]).result()

    print(result[0].data.evs, result[0].data.stds)
    """

    #problem 3
    """
    bs1 =  QuantumCircuit(2) #create the 5 circuits
    bs2=QuantumCircuit(2)
    bs3=QuantumCircuit(2)
    bs4=QuantumCircuit(2)
    jh=QuantumCircuit(2)

    bs1.h(0)
    bs1.cx(0,1)

    bs2.x(0)
    bs2.h(0)
    bs2.cx(0,1)

    bs3.x(1)
    bs3.h(0)
    bs3.cx(0,1)

    bs4.x(0)
    bs4.x(1)
    bs4.h(0)
    bs4.cx(0,1)

    #unentangled hadamare state
    jh.h(0)
    jh.h(1)

    #print diagrams
    print(bs1.draw("text"))
    print(bs2.draw("text"))
    print(bs3.draw("text"))
    print(bs4.draw("text"))
    print(jh.draw("text"))

    circuits = [bs1, bs2, bs3, bs4, jh]
    for qc in circuits:
        qc.measure_all()
    #run on hardware
    counts = run_on_hardware(circuits,10000,DEFAULT_BACKEND,"default")
    

    plot_histogram(counts[0])#graph
    plot_histogram(counts[1])
    plot_histogram(counts[2])
    plot_histogram(counts[3])
    plot_histogram(counts[4])
    plt.show()

    service = QiskitRuntimeService()
    backend = service.backend(DEFAULT_BACKEND)
    
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_qc = pm.run(circuits)#transpiles
    labels = ["II", "IX", "XI", "IZ", "ZI", "XZ", "ZX", "XX", "ZZ"]
    observables = [convertStringToPauli(p) for p in labels]
    pubs = [#transpiles observables
        (qc, [o.apply_layout(qc.layout) for o in observables])
        for qc in isa_qc          #
    ]

    est = RuntimeEstimator(mode=backend)
    est.options.default_shots = 1000
    result = est.run(pubs).result()


    names=["bs1", "bs2", "bs3", "bs4", "jh"]
    for name, pub_result in zip(names, result):
        plt.plot(labels, pub_result.data.evs, marker="o", label=name)

    plt.legend()
    plt.ylabel("expectation value")
    plt.show()
    """
    return 0

if __name__ == "__main__":
    main()  