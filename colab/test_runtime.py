"""CPU-only contracts; run on Colab, without loading any model."""
import unittest
from unittest.mock import patch
import runtime


class Contracts(unittest.TestCase):
    def test_t4_fails_before_install(self):
        with patch.object(runtime, 'inspect_environment', return_value={'generation_blockers':['T4 unsupported']}), patch.object(runtime, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'T4 unsupported'):
                runtime.setup()
            run.assert_not_called()

    def test_cost_includes_failed_time(self):
        r = runtime.cost_report(3600, 50, 2, 100)
        self.assertEqual(r['estimated_yen_per_success'], 4)
        self.assertFalse(r['target_met'])
        self.assertEqual(r['seconds_budget_per_success'], 18)

    def test_unknown_price_is_not_free(self):
        self.assertIsNone(runtime.cost_report(100, 1, 0, 0)['target_met'])
        self.assertIsNone(runtime.cost_report(100, 0, 2, 10)['estimated_yen_per_success'])
        with self.assertRaises(ValueError):
            runtime.cost_report(float('nan'), 1, 2, 10)

    def test_graph_reaches_vsa(self):
        g = runtime.graph('A car', 1)
        self.assertEqual(g['10']['inputs']['model'], ['6',0])
        self.assertEqual(g['11']['inputs']['model'], ['6',0])
        self.assertEqual(g['6']['class_type'], 'H3VSA')
        self.assertEqual(g['10']['inputs']['steps'], 4)
        self.assertEqual(g['9']['inputs']['sampler_name'], 'euler')
        for node in g.values():
            for value in node['inputs'].values():
                if isinstance(value, list):
                    self.assertIn(value[0], g)

    def test_invalid_geometry(self):
        for values in ({'width':607}, {'frames':125}, {'height':0}):
            with self.assertRaises(ValueError):
                runtime.graph('test', 1, **values)

    def test_local_guard(self):
        with patch.object(runtime.sys, 'platform', 'win32'):
            with self.assertRaisesRegex(RuntimeError, 'local PC'):
                runtime.require_colab()


if __name__ == '__main__':
    unittest.main()
