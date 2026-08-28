import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main"))

from FNN_numpy import MLP


class FlexiNNTest(unittest.TestCase):
    def test_adam_learns_xor(self):
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        targets = np.eye(2, dtype=np.float32)[[0, 1, 1, 0]]
        network = MLP(
            2,
            [8],
            2,
            learning_rate=0.03,
            learning_time=1000,
            activation="leaky_relu",
            optimizer="adam",
            seed=7,
            print_output=False,
        )

        network.train(inputs, targets)

        self.assertEqual(network.accuracy(network.predict(inputs), targets), 1.0)

    def test_batch_norm_statistics_survive_save_and_load(self):
        rng = np.random.RandomState(3)
        inputs = rng.randn(16, 3).astype(np.float32)
        targets = np.eye(2, dtype=np.float32)[rng.randint(0, 2, size=16)]
        network = MLP(
            3,
            [4],
            2,
            learning_rate=0.01,
            optimizer="adam",
            batch_norm=True,
            seed=5,
            print_output=False,
        )
        network.train(inputs, targets, batch_size=4, epochs=3)
        expected = network.predict(inputs)

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model"
            network.save(model_path)
            restored = MLP(3, [4], 2, seed=11)
            restored.load(model_path)

        np.testing.assert_allclose(restored.predict(inputs), expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
