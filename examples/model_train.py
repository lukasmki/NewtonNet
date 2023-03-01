import os
import torch

import numpy as np

from torch.optim import Adam
from newtonnet.data import parse_train_test_no_config
from newtonnet.models import NewtonNet
from newtonnet.train import Trainer

torch.set_default_tensor_type(torch.DoubleTensor)

# device = [torch.device("cpu")]  # cpu
device = [
    torch.device(item) for item in ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
]  # gpu

# new model
# model = NewtonNet(device[0], atomic_properties=False, pair_properties=True)

# load from saved
model_path = "./path/to/saved/model.pt"
model = torch.load(model_path)

# optimizer
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = Adam(trainable_params, lr=0.001, weight_decay=0)

# set w_a = 0, w_p = 0 to disable atomic or pair property backprop
def loss(preds, batch_data, w_e, w_f, w_a=0, w_p=0):
    err_sq = 0

    diff_en = preds["E"] - batch_data["E"]
    err_sq += w_e * torch.mean(diff_en**2)

    diff_fc = preds["F"] - batch_data["F"]
    err_sq += w_f * torch.mean(diff_fc**2)

    if w_a > 0:  # Atomic Property
        diff_a = preds["Ai"] - batch_data["Ai"]
        err_sq += w_a * torch.mean(diff_a**2)

    if w_p > 0:  # Pair Property
        diff_p = preds["Pij"] - batch_data["Pij"]
        err_sq += w_p * torch.mean(diff_p**2)

    return err_sq


# init trainer
trainer = Trainer(
    device=device,
    model=model,
    loss_fn=loss,
    optimizer=optimizer,
    energy_loss_w=0.0,  # energy loss fn weight
    force_loss_w=0.0,  # force loss fn weight
    lr_scheduler=["decay", 0.345387763],
    yml_path="",
    checkpoint_test=1,
    checkpoint_model=1,
    output_path="./output/",
    path_iter=1,
    script_name="train_all.py",
    hooks={"vismolvector3d": False},
)

# load data
train_path = "../example_data.npz"
training_data = dict(np.load(train_path, allow_pickle=True))  # get nframes
nframes = training_data["N"].shape[0]

(
    train_gen,
    val_gen,
    test_gen,
    tr_steps,
    val_steps,
    test_steps,
    normalizer,
) = parse_train_test_no_config(
    device=device[0],
    train_path=train_path,
    train_size=nframes - 4,
    val_size=2,
    test_size=2,
    train_batch_size=10,
    val_batch_size=10,
)

trainer.train(
    epochs=10,
    train_generator=train_gen,
    steps=tr_steps,
    val_generator=val_gen,
    val_steps=val_steps,
    test_generator=test_gen,
    test_steps=test_steps,
    clip_grad=1.0,
)
