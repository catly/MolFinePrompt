import json
import os
import torch
import torch.nn as nn
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

class promSmiles(torch.nn.Module):
    def __init__(self, num_classes, pretrained_model_file, device):
        super(promSmiles, self).__init__()
        self.premodel = pre_trained(pretrained_model_file, device)
        premodel_state = torch.load('result/pretrain/model_state.pt')
        self.premodel.load_state_dict(premodel_state)
        print("model state done!")
        self.device = device
        self.gmodel = self.premodel.graph_encoder

        self.peft_config = PromptTuningConfig(
            task_type="PROMPT_SMILES",
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=100,
            num_transformer_submodules=1,
            prompt_tuning_init_text=task_prompt_text,
            tokenizer_name_or_path="scibert_scivocab_uncased",
        )

        self.ptmodel = get_peft_model(self.premodel.text_encoder, self.peft_config)
        self.pred_linear = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, (self.emb_dim) // 2),
            torch.nn.ReLU(),
            torch.nn.Linear((self.emb_dim) // 2, num_classes)

        )
    def encode_text(self, data):
        return self.ptext_encoder.encode_text(data)
    def encode_graph(self, data):
        return self.premodel.graph_encoder(data)

    def forward(self, graph_data1, graph_data2):
        prompt = self.ptmodel(batch_size=len(graph_data1), device=self.device)
        graph_data_rep1 = self.encode_graph(graph_data1).to(prompt.device)
        graph_data_rep2 = self.encode_graph(graph_data2).to(prompt.device)
        graph_data_rep1 = prompt + graph_data_rep1
        multidata_rep = torch.cat((graph_data_rep1, graph_data_rep2), dim=1)
        return self.pred_linear(multidata_rep)