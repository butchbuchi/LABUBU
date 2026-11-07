import unittest
import tempfile
from pathlib import Path
import numpy as np
from labubu import optimize, read_qasm, entanglement_cost
from labubu.scheduler import repair, sweep
from labubu.encoding import FT_circuit_func
import random


class SchedulingTests(unittest.TestCase):
    def test_cost_matches_hand_count(self):
        g = np.array([[1, 0, -1], [-1, 2, 1]])
        m = np.array([[0, 1, 1], [0, 0, 1]])
        self.assertEqual(entanglement_cost(g, m), {"total": 3, "telegate": 2, "teledata": 1})

    def test_rollback_cascade_and_fallback(self):
        current = [0, 0, 1, 1, 2, 2]
        proposal = [1, 0, 2, 2, 2, 2]
        self.assertEqual(repair(proposal, current, [2, 2, 2], 1), current)
        repaired = repair(proposal, current, [2, 2, 2], 500)
        self.assertTrue(all(repaired.count(i) <= 2 for i in range(3)))
        self.assertEqual(repair(proposal, current, [2, 2, 2], 500), repaired)

    def test_returned_mapping_cost_capacity_and_seed(self):
        g = np.array([[1, 0, 3, 2], [2, 3, 0, 1], [-1, -1, -1, -1]])
        for seed in range(18):
            a = optimize(g, [3, 3], iterations=230, interval=5, seed=seed)
            b = optimize(g, [3, 3], iterations=230, interval=5, seed=seed)
            self.assertEqual(a, b)
            self.assertEqual(entanglement_cost(g, np.array(a["mapping"]))["total"], a["total"])
            self.assertEqual(min(x["cost"] for x in a["checkpoints"]), a["total"])
            self.assertTrue(all(max(np.bincount(row)) <= 3 for row in a["mapping"]))

    def test_initialization_anchor(self):
        from labubu.scheduler import initialize
        g = np.array([[-1, 2, 1, 4, 3, -1]])
        mapping = initialize(g, [5, 5], 1)
        self.assertEqual(mapping.tolist(), [[0, 0, 0, 1, 1, 0]])
        self.assertTrue(all(np.count_nonzero(mapping[0] == i) <= 4 for i in range(2)))

    def test_update_schedule(self):
        from unittest.mock import patch
        with patch("labubu.scheduler.sweep") as mocked:
            result = optimize([[1, 0]], [3], iterations=1000, interval=50)
        self.assertEqual(mocked.call_count, 1000)
        bp_epochs = [i for i, call in enumerate(mocked.call_args_list)
                     if call.args[4]]
        self.assertEqual(bp_epochs, list(range(100, 900, 50)))
        self.assertEqual([c["before_iteration"] for c in result["checkpoints"][:-1]],
                         [i + 1 for i in bp_epochs])
        for budget in [1, 30, 100, 150, 200]:
            with patch("labubu.scheduler.sweep") as mocked:
                optimize([[1, 0]], [3], iterations=budget)
            self.assertEqual(mocked.call_count, budget)
            self.assertFalse(any(c.args[4] for c in mocked.call_args_list))

    def test_one_layer_and_invalid_inputs(self):
        self.assertEqual(optimize([[1, 0]], [3], iterations=2)["total"], 0)
        with self.assertRaises(ValueError):
            optimize([[1, -1]], [3])
        with self.assertRaises(ValueError):
            optimize([[1, 0]], [1], iterations=2)

    def test_qasm_header_and_layer_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "test.qasm"
            p.write_text('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n')
            self.assertEqual(read_qasm(p).tolist(), [[-1, -1], [1, 0]])

    def test_steane_interactions(self):
        encoded = FT_circuit_func(np.array([[1, 0]]), 5)
        self.assertEqual(encoded.shape, (84, 16))
        for row in encoded:
            for q, p in enumerate(row):
                if p >= 0:
                    self.assertEqual(row[p], q)

    def test_cpg_is_not_globally_monotone(self):
        # Explicit counterexample: simultaneous proposals can increase cost.
        g = np.array([[1, 0, -1, -1]] * 3)
        m = np.array([[0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
        before = entanglement_cost(g, m)["total"]
        rng = random.Random(78)
        for _ in range(12):
            rng.randrange(2)  # recreate the state used to generate this mapping
        sweep(g, m, [4, 4], rng)
        self.assertEqual((before, entanglement_cost(g, m)["total"]), (8, 9))


if __name__ == "__main__":
    unittest.main()
