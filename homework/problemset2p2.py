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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00-Setup-Start-Here"))
from _job_state import (
    DEFAULT_POLL_INTERVAL,
    extract_counts,
    save_last_job,
    wait_for_job,
)



def convertObservableToPauli(M: np.ndarray) -> SparsePauliOp:
    """takes in a 2d matrix, converts to pauli"""
    op = Operator(M) #converts to quiskit operator
    return SparsePauliOp(op) #converts to pauli expression 
DEFAULT_BACKEND = "ibm_rensselaer"

#copyed from provided files
def run_on_hardware(
    circuit: QuantumCircuit,
    shots: int,
    backend_name: str,
    account_name: str,
    preferred_instance: str = "MANE-4960-dedicated",
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> dict | None:
    """Run a quantum circuit on IBM hardware via Qiskit Runtime.

    Transpiles the circuit for the target backend, submits the job, and waits
    for results. The job ID is saved to disk for recovery if interrupted.

    Parameters
    ----------
    circuit : QuantumCircuit
        The quantum circuit to run.
    shots : int
        Number of shots (repetitions) for sampling.
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
    dict or None
        Dictionary mapping bitstrings to counts if successful, None if the
        connection failed or the job was interrupted.
    """
    print(
        f"\n--- Running on IBM hardware backend '{backend_name}', "
        f"{shots} shots ---"
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

    # Extra step, only needed for real hardware: convert to an ISA
    # (Instruction Set Architecture) circuit that is transpiled into this
    # backend's actual basis gates and qubit connectivity. The simulator
    # above skipped this because it supports every gate directly and has
    # no connectivity restrictions.
    transpile_start = time.perf_counter()
    pass_manager = generate_preset_pass_manager(
        backend=backend, optimization_level=1
    )
    isa_circuit = pass_manager.run(circuit)
    transpile_time = time.perf_counter() - transpile_start
    print(f"Transpiled circuit in {transpile_time:.2f}s")

    # Steps 4-5 again, this time targeting real hardware: same PUB shape,
    # but the sampler now runs in "backend mode" against the IBM backend.
    pub = (isa_circuit,)
    sampler = RuntimeSampler(mode=backend)

    # Step 6: submit, then retrieve. Save the job ID to disk *before*
    # waiting. The queue can be long, and the job keeps running on IBM's
    # servers even if this script's connection drops while it waits.
    job = sampler.run([pub], shots=shots)
    save_last_job(job.job_id(), backend.name, account_name)
    print(f"Submitted job {job.job_id()} on {backend.name}.")
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

    # Step 7: post-process, exactly as with the Aer result above.
    pub_result = result[0]
    _, counts = extract_counts(pub_result)

    elapsed = time.perf_counter() - start_time
    print(f"\n*** Hardware job completed! ***")
    print(f"Counts: {counts}")
    print(f"Total elapsed time: {elapsed:.2f}s")
    return counts

def main() -> int:
    qc1 = qc2 = qc3 = prepQubitRotation(np.pi/4,np.pi/3,np.pi/6) #prep qubits
    measureInGivenAxis(qc1,'X') #measure in all 3 axis
    measureInGivenAxis(qc2,'Y')
    measureInGivenAxis(qc3,'Z') 

    counts1 = run_on_hardware(qc1,10000,DEFAULT_BACKEND,"default")# run on hardware
    counts2 = run_on_hardware(qc2,10000,DEFAULT_BACKEND,"default")
    counts3 = run_on_hardware(qc3,10000,DEFAULT_BACKEND,"default")

    plot_histogram(counts1)#graph
    plot_histogram(counts2)
    plot_histogram(counts3)
    plt.show()
    return 0

if __name__ == "__main__":
    main()  