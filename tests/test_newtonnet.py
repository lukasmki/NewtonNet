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
        mij = torch.ones((10, 6, 6, 128))
        m = PairProperty(128, torch.sigmoid)
        pred = m.forward(mij)
        assert pred.shape == torch.Size([10, 6, 6, 1])

class TestNewtonNet(unittest.TestCase):
    def test_forward(self):
        data = {
            'Z' : torch.ones(10, 6, dtype=torch.long),
            'R' : torch.rand(10, 6, 3),
            'N' : torch.ones(10, 1) * 6,
            'NM': torch.ones(10, 6, 5),
            'AM': torch.ones(10, 6),
        }
        m = NewtonNet(20, 128, torch.sigmoid, atomic_properties=True, pair_properties=True)
        result = m.forward(data)


if (__name__=="__main__"):
    unittest.main()