import logging
import os 
import torch
import argparse
import torch.nn as nn

from torch import optim
from network.slan import SLAN
from dataset.soba import SOBA
from tools.trainer import *

amp = True

paths = {
    "sample_path" : "./data/imgs",
    "object_label_path" : "./data/object_masks",
    "shadow_label_path" : "./data/shadow_masks",
    
    "test_sample_path" : "./data/SBU-test/test_imgs",
    "test_object_label_path" : "./data/SBU-test/test_object_masks",
    "test_shadow_label_path" : "./data/SBU-test/test_shadow_masks",

    "train_light_path" : "./data/light_annotations.txt",
    "test_light_path" : "./data/SBU-test/test_light_annotations.txt",
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 5, help = "training epochs")
    parser.add_argument("--batch_size", type = int, default = 1, help = "training batch size")
    parser.add_argument("--learning_rate", type = float, default = 1e-5, help = "training learning late")
    parser.add_argument("--weight_decay", type = float, default = 1e-8, help = "training weight decay")
    parser.add_argument("--val_step", type = int, default = 1, help = "validation epoch")
    parser.add_argument("--checkpoint_step", type = int, default = 5, help = "checkpoint file save step")
    parser.add_argument("--checkpoint_path", type = str, default = "./checkpoint", help = "checkpoint file path")

    return parser.parse_args()

def train_net(model) :
    train_set = SOBA(paths["sample_path"], paths["object_label_path"], paths["shadow_label_path"], paths["train_light_path"])
    val_set = SOBA(paths["test_sample_path"], paths["test_object_label_path"], paths["test_shadow_label_path"], paths["test_light_path"])

    loader_args = dict(batch_size = args.batch_size, num_workers = os.cpu_count(), pin_memory = True)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle = True, **loader_args)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle = False, drop_last = True, **loader_args)

    optimizer = optim.RMSprop(model.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay, momentum = 0.999, foreach = True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience = 5)  
    grad_scaler = torch.cuda.amp.GradScaler(enabled = amp)
    criterion = nn.CrossEntropyLoss() 

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        args.val_step,
        optimizer,
        criterion,
        device,
        scheduler,
        grad_scaler,
        args.checkpoint_step,
        args.checkpoint_path,
    )
    trainer.run(args.epochs)

if (__name__ == "__main__") :
    args = get_args()
    logging.basicConfig(level = logging.INFO, format="[%(levelname)s] : %(message)s")
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    logging.info(f"using device {device}")
    model = SLAN(num_classes = 2, in_channels = 3).to(device)
    train_net(model)