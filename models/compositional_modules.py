import os
import torch
import torch.nn as nn

from models.drug_pertain import get_drug
from models.graph import GNN_finetune
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def get_model(config, device):
    interface = get_drug(config, device)

    optimizer = torch.optim.Adam(
        [
        {"params": interface.logit_scale, 'lr': config.logit_scale_lr},
        {"params": interface.text_encoder.parameters(), 'lr': config.lr},
         {"params": interface.graph_encoder.model.gnn.parameters(),'lr': config.glr}
        ], weight_decay=config.weight_decay
    )

    return interface, optimizer


