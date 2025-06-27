import json
import os
import torch
import torch.nn as nn
import yaml
from models.drug_pertain import get_drug, scibert_load
from modules.sci_bert import ScibertEncoder
from prompt import (
    PeftModel,
    PeftConfig,
    PromptTuningLoRAConfig,
    get_peft_model,
    PromptTuningInit,
    PromptTuningConfig,
    TaskType,
)
from transformers import PreTrainedModel
class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

    def __getattr__(self, name):
        return self.__dict__.get(name)
def pre_trained(pretrained_model_filepath, device):
    with open(os.path.join(pretrained_model_filepath, 'config.yaml'), "rb") as fp:
        loaded_config = yaml.safe_load(fp)

    loaded_config = Config(loaded_config)

    model = get_drug(loaded_config, device)
    return model

class retmodel(torch.nn.Module):
    def __init__(self, num_classes, config, device):
        super(retmodel, self).__init__()
        self.premodel = torch.load(os.path.join(config.pretrained_model_filepath, 'final_model.pt'))
        model_state = torch.load(os.path.join(config.pretrained_model_filepath, 'model_state.pt'))
        self.premodel.load_state_dict(model_state)
        for param in self.premodel.parameters():
            param.requires_grad = False
        self.graph_lr = nn.Linear(768, 256)
        self.text_lr = nn.Linear(768, 256)
        self.device = device
    def encode_text(self, data):
        text_features = self.premodel.text_encoder.encode_text(data).to(self.device)
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )
        text_data_rep = self.text_lr(text_features)
        return text_data_rep
    def encode_graph(self, data):
        graph_data_rep = self.premodel.encode_graph(graph_data).to(self.device)
        
        graph_data_rep = self.graph_lr(graph_data_rep)
        return graph_data_rep
    def forward(self, text_data, graph_data):
       
        graph_data_rep = self.encode_graph(graph_data)
        text_data_rep = self.encode_text(text_data)

        multidata_rep = torch.cat((prompt, graph_data_rep), dim=1)
        return self.pred_linear(multidata_rep)