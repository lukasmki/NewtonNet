import unittest
import torch
import numpy as np
from newtonnet.models.newtonnet import AtomicProperty, PairProperty, NewtonNet


class TestAtomicProperty(unittest.TestCase):
    def test_forward(self):
        a = torch.ones((10, 6, 128))
        m = AtomicProperty(128, torch.sigmoid, 0.0)
        pred = m.forward(a)
        assert pred.shape == torch.Size([10, 6, 1])


class TestPairProperty(unittest.TestCase):
    def test_forward(self):
        mij = torch.ones((10, 6, 5, 128))
        m = PairProperty(128, torch.sigmoid)
        pred = m.forward(mij)
        assert pred.shape == torch.Size([10, 6, 5, 1])


if __name__ == "__main__":
    unittest.main()
