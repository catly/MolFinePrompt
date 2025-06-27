import argparse
import torch
import torch.nn as nn
# from clip.model import CLIP
# from .text_encoder import Cu
# stomTextEncoder
from .sci_bert import ScibertEncoder
from .graph_encoder import GraphEncoder
from graph.data_utils import *


class ModuleInterface(torch.nn.Module):
    def __init__(
        self,
        tokenizer,
        text_model,
        graph_model,
        config: argparse.ArgumentParser,
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cuda:0",
    ):

        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        if dtype is None and device == "cpu":
            self.dtype = torch.float32
        elif dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype

        self.device = device

        self.graph_encoder = GraphEncoder(graph_model, self.device)
        self.text_encoder = ScibertEncoder(self.tokenizer, text_model, self.device)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def encode_text(self, text):
        return self.text_encoder.encode_text(
            text
        )

    def GraphEncoder(self, graph_data, gmodel):
        graph_batch = molgraph_to_graph_data(graph_data)
        graph_batch = graph_batch.to(self.config.device)
        super_node_rep = gmodel(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.num_part)
        return super_node_rep
    
    def forward(self, text_data, graph_data):
        text_features = self.encode_text(text_data)

        graph_features = self.graph_encoder(graph_data)
        idx_text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )
        normalized_img = graph_features / graph_features.norm(dim=-1, keepdim=True)

        logits = (
            self.logit_scale.exp()
            * idx_text_features
            @ normalized_img.t()
        )

        return logits, idx_text_features, normalized_img