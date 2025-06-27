import os

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import transformers
from transformers import AutoTokenizer, AutoModel
# from clip_modules.model_loader import load

# from clip_modules.interface import Interface
from modules.interface import *
from models.graph import GNN_finetune

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class DrugInterface(ModuleInterface):
    def __init__(
        self,
        tokenizer,
        text_model,
        graph_model,
        config,
        device,
    ):
        super().__init__(tokenizer, text_model, graph_model, config, device=device)

def scibert_load():
    model_name = "./scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def drug_init(
    config,
    device
):

    tokenizer, text_model = scibert_load()
    graph_model = GNN_finetune(config.num_layer, config.emb_dim, JK=config.JK, drop_ratio=config.dropout_ratio,
                               gnn_type=config.gnn_type)
    if not config.graph_pretrain_file == "":
        graph_model.from_pretrained(config.graph_pretrain_file, device)
    return tokenizer, text_model, graph_model



def get_drug(config, device):

    tokenizer, text_model, graph_model = drug_init(config, device)
    interface = DrugInterface(tokenizer, text_model, graph_model, config, device)
    return interface

