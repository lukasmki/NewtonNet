import os
import torch

import numpy as np

from torch.optim import Adam
from newtonnet.data import parse_train_test_no_config
from newtonnet.data.neighbors import ExtensiveEnvironment
from newtonnet.data.loader import extensive_train_loader
from newtonnet.models import NewtonNet
from newtonnet.train import Trainer

torch.set_default_tensor_type(torch.DoubleTensor)

# device = [torch.device("cpu")]  # cpu
device = [
    torch.device(item) for item in ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
]  # gpu

model = torch.load("./bo-model/training_3/models/best_model.pt")  # load from saved
model.to(device[0])
model.eval()

DATA_PATH = "./dataset/"
pred_files = [x for x in os.listdir(DATA_PATH)]

# train aimd
for file in sorted(pred_files):

    data = dict(np.load(f"{DATA_PATH}/{file}", allow_pickle=True))
    num_frames = data["N"].shape[0]

    batch_gen = extensive_train_loader(
        data,
        env_provider=ExtensiveEnvironment(),
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    # evaluate model
    output = dict()
    for s in range(num_frames):
        # get new batch and ensure tensors on same device
        batch_data = next(batch_gen)
        for k, v in batch_data.items():
            batch_data[k] = v.to(device[0])

        # prediction!
        result = model.forward(batch_data)

        # collect input data
        for k in ["R", "Z", "N"]:
            val = batch_data[k].detach().cpu().numpy()
            if k not in output:
                output[k] = val
            else:
                output[k] = np.concatenate((output[k], val), axis=0)

        # concatenate results
        for k, v in result.items():
            if k == "hs":  # skip latent forces
                continue
            val = v.detach().cpu().numpy()
            if k not in output:
                output[k] = val
            else:
                output[k] = np.concatenate((output[k], val), axis=0)

        print(f"\r{(s+1)/num_frames * 100:0.2f}%, {s+1}/{num_frames}", end="\r")
    np.savez_compressed("./evaluate/" + file + "-output.npz", **output)
    print("\n")
