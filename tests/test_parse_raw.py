import unittest
import torch
from newtonnet.data.parse_raw import parse_train_test

class TestParseTrainTest(unittest.TestCase):
    def setUp(self):
        self.settings = {
            'general': {
                'driver': 'run.py'
            },
            'data': {
                'train_path':'./tests/example_data.npz',
                'test_path' : False,
                'train_size': 1000,
                'val_size'  : 100,
                'test_size' : 100,
                'random_states' : 90,
            },
            'training': {
                'tr_batch_size' : 12,
                'tr_frz_rot' : False,
                'tr_rotations': 0,
                'tr_keep_original': True,
                'val_batch_size': 12,
                'val_frz_rot' : False,
                'val_rotations': 0,
                'val_keep_original': True,
                'shuffle' : False,
                'drop_last' : True,
            },
            }
        self.device = torch.device('cpu')
    
    def test_parse_train_test(self):
        train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps, normalizer = parse_train_test(self.settings, self.device)
        for batch in test_gen:
            print(batch.keys(), [batch[k].shape for k in batch.keys()])
            print(batch['NM'][0])
            print(batch['AM'][0])
            break

if (__name__=="__main__"):
    unittest.main()