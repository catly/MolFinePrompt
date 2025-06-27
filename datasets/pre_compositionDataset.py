import os
import codecs
import csv
import torch
from torch.utils.data import Dataset
from graph.data_utils import MoleculeDataset
from graph.chemutils import get_mol

class preCompositionDataset(Dataset):
    def __init__(
            self,
            root,
            phase,
            type = 'pretrain'
    ):
        self.root = root
        self.phase = phase
        self.type = type
        self.train_data = self.get_split_info()

        self.text_data = [data[0] for data in self.train_data]
        self.graph_data = [data[1] for data in self.train_data]

        self.graph_mol_data = MoleculeDataset(self.graph_data)

        self.data = [[text, graph] for text, graph in zip(self.text_data, list(self.graph_mol_data))]

    def get_split_info(self):
        with codecs.open(os.path.join(self.root, file), 'r',encoding='utf-8') as f:
            data = csv.DictReader(f, skipinitialspace=True)
            train_data= []

            for instance in data:
                smiles, text = instance['smiles'], instance['description']
                if smiles != '*':
                    mol = get_mol(smiles)
                    if mol is None:
                        continue
                    else:
                        data_i = [text, smiles]
                        train_data.append(data_i)
        return train_data

    def parse_split(self):
        text, graph = zip(*self.train_data)
        return text, graph

    def __getitem__(self, index):
        text, smiles = self.data[index]
        data = [
            text, smiles, index
        ]
        return data

    def __len__(self):
        return len(self.data)


