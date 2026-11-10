#!/usr/bin/env python3
"""Build a Bell state and run it on a simulator, then on real IBM hardware.

A guided walkthrough of the modern Qiskit Runtime workflow: named
registers, a named circuit, PUBs, samplers, job submission/retrieval, and
post-processing raw shot data into counts. No plotting here.

The hardware queue can be long, so while waiting this script prints a
status/queue update every few seconds (see --poll-interval). The job ID is
also saved locally as soon as it's submitted -- if this script loses its
connection (or you close your laptop) while waiting, the job keeps running
on IBM's servers regardless; fetch its result later with
05_retrieve_job.py instead of resubmitting.

Run with no arguments. Pass --skip-hardware to run only the simulator step.
"""

import argparse
import logging
import sys
import time

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler

from _job_state import (
    DEFAULT_POLL_INTERVAL,
    extract_counts,
    save_last_job,
    wait_for_job,
)

logging.basicConfig(
    level=logging.WARN,
    format="[%(levelname)-8s] %(name)s: %(message)s",
)
# Suppress harmless Qiskit warnings about instance discovery
logging.getLogger("qiskit_runtime_service").setLevel(logging.ERROR)

DEFAULT_SHOTS = 10000
DEFAULT_BACKEND = "ibm_rensselaer"
PREFERRED_INSTANCE = "MANE-4960-dedicated"


def build_bell_circuit() -> QuantumCircuit:
    """Build a Bell state quantum circuit with named registers.

    Constructs a two-qubit circuit that creates an entangled Bell state
    (|00⟩ + |11⟩)/√2 and measures both qubits.

    Returns
    -------
    QuantumCircuit
        A named Bell state circuit with 2 qubits and 2 classical bits.
    """
    # Step 1: Create named registers. Naming them makes the circuit diagram and
    # the post-processing results (pub_result.data.<name>) self-explanatory
    # instead of falling back to generic names like "q" and "c0".
    qubits = QuantumRegister(4, name="q")
    bits = ClassicalRegister(4, name="meas")

    # Step 2: Create a named circuit, built from those registers.
    circuit = QuantumCircuit(qubits, bits, name="bell_state")

    # Step 3: Create the Bell state itself. H puts qubit 0 into superposition,
    # then CX entangles it with qubit 1.
    circuit.x(0)
    circuit.swap(qubits[0],qubits[1])
    circuit.swap(qubits[1],qubits[2])
    circuit.swap(qubits[2],qubits[3])
    circuit.measure(qubits, bits)

    return circuit


def run_on_aer(circuit: QuantumCircuit, shots: int) -> dict:
    """Run a quantum circuit on the Aer simulator.

    Parameters
    ----------
    circuit : QuantumCircuit
        The quantum circuit to simulate.
    shots : int
        Number of shots (repetitions) for sampling.

    Returns
    -------
    dict
        Dictionary mapping bitstrings to their measurement counts.
    """
    print(
        f"\n--- Running on a perfect (noiseless) Aer simulator, "
        f"{shots} shots ---"
    )
    start_time = time.perf_counter()

    # Step 4: a PUB (Primitive Unified Bloc) is what you hand to a sampler.
    # A PUB is a tuple of (circuit, [parameter values], [shots]) -- here
    # just the circuit, since the Bell circuit has no parameters.
    pub = (circuit,)

    # Step 5: create the sampler. qiskit_aer.primitives.SamplerV2 samples
    # locally against an ideal, noiseless simulator by default -- no real
    # hardware noise, so counts should split ~50/50 between "00" and "11".
    sampler = AerSampler()

    # Step 6: submit the job, then retrieve its result.
    job = sampler.run([pub], shots=shots)
    print(f"Submitted job {job.job_id()}")
    result = job.result()

    # Step 7: post-process. pub_result.data.<classical register name> is a
    # BitArray of raw shot outcomes; get_counts() collapses it into a
    # bitstring -> count dictionary.
    pub_result = result[0]
    _, counts = extract_counts(pub_result)

    elapsed = time.perf_counter() - start_time
    print(f"\n*** Simulator job completed! ***")
    print(f"Counts: {counts}")
    print(f"Elapsed time: {elapsed:.2f}s")
    return counts


def run_on_hardware(
    circuit: QuantumCircuit,
    shots: int,
    backend_name: str,
    account_name: str,
    preferred_instance: str = PREFERRED_INSTANCE,
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


def build_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser for running quantum circuits.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build a Bell state and run it on Aer, then on IBM hardware."
        ),
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help=f"Shots per run (default: {DEFAULT_SHOTS}).",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help=f"IBM backend to run on (default: {DEFAULT_BACKEND}).",
    )
    parser.add_argument(
        "--name",
        default="default",
        help="Name of the saved account to use (default: 'default').",
    )
    parser.add_argument(
        "--skip-hardware",
        action="store_true",
        help="Only run on the Aer simulator.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help=(
            f"Seconds between status/queue checks while waiting on "
            f"hardware (default: {DEFAULT_POLL_INTERVAL})."
        ),
    )
    return parser


def main() -> int:
    """Build a Bell state and run it on a simulator, then on real IBM hardware.

    Returns
    -------
    int
        Exit code: always 0 (failures are reported but don't block completion).
    """
    parser = build_parser()
    args = parser.parse_args()
    overall_start = time.perf_counter()

    circuit_start = time.perf_counter()
    circuit = build_bell_circuit()
    circuit_time = time.perf_counter() - circuit_start
    print(f"Circuit built in {circuit_time:.2f}s")
    print("Circuit:")
    print(circuit.draw(output="text"))

    aer_counts = run_on_aer(circuit, args.shots)

    hardware_counts = None
    if not args.skip_hardware:
        hardware_counts = run_on_hardware(
            circuit,
            args.shots,
            args.backend,
            args.name,
            poll_interval=args.poll_interval,
        )

    overall_time = time.perf_counter() - overall_start
    print("\n=== Summary ===")
    print(f"Aer (ideal) counts:       {aer_counts}")
    if hardware_counts is not None:
        print(f"{args.backend} counts: {hardware_counts}")
    elif not args.skip_hardware:
        print(f"{args.backend}: run failed or was unreachable (see above).")
    print(f"\nTotal program time: {overall_time:.2f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
