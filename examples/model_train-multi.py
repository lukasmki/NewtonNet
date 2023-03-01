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

# model = NewtonNet(device[0], pair_properties=True)  # new model
model_path = "./model/training_20/models/best_model.pt"
model = torch.load(model_path)  # load from saved

# optimizer
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = Adam(trainable_params, lr=0.001, weight_decay=0)


def loss(preds, batch_data, w_e, w_f, w_a=0, w_p=20):
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


DATA_PATH = "./dataset/"
train_files = [x for x in os.listdir(DATA_PATH)]

# train aimd
for file in sorted(train_files):
    print(file)
    training_data = dict(np.load(f"{DATA_PATH}/{file}", allow_pickle=True))
    nframes = training_data["N"].shape[0]

    # init trainer
    trainer = Trainer(
        device=device,
        model=model,
        loss_fn=loss,
        optimizer=optimizer,
        energy_loss_w=0.0,
        force_loss_w=0.0,
        lr_scheduler=["decay", 0.345387763],
        yml_path="",
        checkpoint_test=1,
        checkpoint_model=1,
        output_path="./output/",
        path_iter=1,
        script_name="train_all.py",
        hooks={"vismolvector3d": False},
    )

    (
        train_gen,
        val_gen,
        test_gen,
        tr_steps,
        val_steps,
        test_steps,
        normalizer,
    ) = parse_train_test_no_config(
        device[0],
        DATA_PATH + file,
        nframes - 4,
        2,
        test_size=2,
        train_batch_size=1,
        val_batch_size=1,
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
