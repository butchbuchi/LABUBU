import tempfile
from pathlib import Path
import unittest
import numpy as np
from labubu import optimize, read_qasm
from labubu.baselines.gate_cover import analyze_qasm


class InterfaceTests(unittest.TestCase):
    def test_integer_validation(self):
        for layers, extra in [([[1.9, .2]], {}), ([[1, 0]], {'initial_mapping': [[0.2, 0.9]]}),
                              ([[1, 0]], {'idle': .5}), ([[1, 0]], {'iterations': True})]:
            with self.assertRaises(ValueError):
                optimize(layers, [3], **extra)

    def test_clifford_guard_and_directed_cx(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'example.qasm'
            p.write_text('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncx q[1],q[0];\n')
            self.assertEqual(read_qasm(p, clifford_only=True).tolist(), [[1, 0]])
            layers, width = analyze_qasm(str(p))
            self.assertEqual((layers.tolist(), width), ([[10001, 0]], 2))
            p.write_text('qreg q[2];\nt q[0];\ncx q[0],q[1];\n')
            with self.assertRaises(ValueError):
                read_qasm(p, clifford_only=True)

    def test_input_not_mutated(self):
        layers = np.array([[1, 0]])
        initial = np.array([[0, 0]])
        optimize(layers, [3], iterations=1, initial_mapping=initial)
        np.testing.assert_array_equal(layers, [[1, 0]])
        np.testing.assert_array_equal(initial, [[0, 0]])
