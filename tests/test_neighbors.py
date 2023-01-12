import unittest
import torch
from newtonnet.data.neighbors import ExtensiveEnvironment

class TestExtensiveEnvironment(unittest.TestCase):
    def testGetEnvironment(self):
        R = torch.rand((10, 6, 3))
        Z = torch.randint(1, 2, (10, 6))

        env = ExtensiveEnvironment()
        neighbors, neighbor_mask, mask, distances, distance_vectors = env.get_environment(R, Z)
        # print(R)
        # print(neighbor_mask)



if (__name__=="__main__"):
    unittest.main()