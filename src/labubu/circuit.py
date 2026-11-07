"""Read the single-register, single-qubit/CX QASM subset used by the project."""
import re
import numpy as np


def read_qasm(path, *, clifford_only=False, directed=False):
    width = None
    clocks = None
    layers = []
    with open(path, encoding="utf-8") as source:
        lines = source.readlines()
    for line in lines:
        line = line.split("//", 1)[0].strip()
        if not line or line.startswith(("OPENQASM", "include", "creg")):
            continue
        match = re.fullmatch(r"qreg\s+(\w+)\[(\d+)\];", line)
        if match:
            if width is not None:
                raise ValueError("Only one quantum register is supported")
            register, width = match[1], int(match[2])
            if width < 1:
                raise ValueError("Circuit width must be positive")
            clocks = [-1] * width
            continue
        if width is None:
            raise ValueError("A qreg declaration must precede gates")
        if line.startswith("barrier"):
            raise ValueError("Barriers require explicit preprocessing")
        op = line.split()[0].split("(")[0]
        if op not in {"cx", "u", "u1", "u2", "u3", "U", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "id"}:
            raise ValueError(f"Unsupported operation: {op}")
        if clifford_only and op not in {'cx', 'h', 'x', 'y', 'z', 's', 'sdg', 'id'}:
            raise ValueError(f'Steane mode requires named Clifford gates; decompose {op} into h/s/Pauli/cx first')
        refs = re.findall(r"(\w+)\[(\d+)\]", line)
        qubits = [int(q) for reg, q in refs]
        if any(reg != register for reg, q in refs) or any(q >= width for q in qubits):
            raise ValueError("Invalid register or qubit")
        if len(qubits) != (2 if op == "cx" else 1) or len(set(qubits)) != len(qubits):
            raise ValueError(f"Invalid gate: {line}")
        tick = max(clocks[q] for q in qubits) + 1
        while len(layers) <= tick:
            layers.append([-1] * width)
        for q in qubits:
            clocks[q] = tick
        if op == "cx":
            a, b = qubits
            if directed and width >= 10000:
                raise ValueError('Legacy directed encoding requires width < 10000')
            layers[tick][a], layers[tick][b] = b, a + 10000 if directed else a
    if width is None:
        raise ValueError("Missing qreg")
    return np.asarray(layers or [[-1] * width], dtype=int)
