import unittest

import numpy as np

from causal_repair.grid import PeriodicGrid
from causal_repair.relaxation import RelaxationParameters, RelaxationSolver


class RelaxationTests(unittest.TestCase):
    def test_exact_homogeneous_decay(self):
        grid = PeriodicGrid((8,), (1.0,), "fd2")
        target = grid.zeros((3,))
        target[1] = 2.0
        parameters = RelaxationParameters(mobility=2.0, susceptibility=0.5)
        solver = RelaxationSolver(grid, lambda _time: target, parameters)
        state = solver.zero_state()
        for _ in range(10):
            state = solver.step(state, 0.1, stochastic=False)
        expected = target * (1.0 - np.exp(-state.time / parameters.tau))
        np.testing.assert_allclose(state.xi, expected, atol=1e-14)

    def test_noise_has_fluctuation_dissipation_variance(self):
        grid = PeriodicGrid((4096,), (4096.0,), "fd2")
        target = grid.zeros((3,))
        parameters = RelaxationParameters(mobility=1.0, susceptibility=0.5, temperature=1.5)
        solver = RelaxationSolver(grid, lambda _time: target, parameters, seed=4)
        state = solver.step(solver.zero_state(), 20.0, stochastic=True)
        expected_variance = parameters.temperature / parameters.susceptibility
        self.assertLess(abs(np.var(state.xi) / expected_variance - 1.0), 0.04)


if __name__ == "__main__":
    unittest.main()

