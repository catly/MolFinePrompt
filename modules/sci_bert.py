import os
import torch
import torchvision.models as models
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModel
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb=512"

device = "cuda:0"
class ScibertEncoder(torch.nn.Module):
    def __init__(self, tokenizer, model, device):
        super(ScibertEncoder, self).__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.dropout = nn.Dropout(0.1)

    def tokenizer_text(self, text, text_max_len):
        sentence_token = self.tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=text_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask

    def encode_text(self, text, text_max_length = 512):

        input_ids, attention_mask = self.tokenizer_text(text,text_max_len=text_max_length)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        text_features = self.forward(input_ids, attention_mask)

        return text_features

    def forward(self, input_ids, attention_mask):
        device = input_ids.device
        typ = torch.zeros(input_ids.shape).long().to(device)
        self.model = self.model.to(device)
        output = self.model(input_ids, token_type_ids=typ, attention_mask=attention_mask)['pooler_output']  # b,d
        logits = self.dropout(output)
        return logits
