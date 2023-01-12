import unittest
import torch
import numpy as np
from newtonnet.models.newtonnet import AtomicProperty, PairProperty

class TestAtomicProperty(unittest.TestCase):
    def testForward(self):
        x = torch.ones((10, 6, 128))
        m = AtomicProperty(128, torch.sigmoid, 0.0)
        pred = m(x)

if (__name__=="__main__"):
    unittest.main()