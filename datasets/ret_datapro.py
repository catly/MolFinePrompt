import codecs
import csv
import os.path
from itertools import product

import numpy as np
import torch
from torch.utils.data import Dataset
from graph.data_utils import MoleculeDataset
from graph.chemutils import get_mol


class retDatapro(Dataset):
    def __init__(
            self,
            root,
            dataset,
            type
    ):
        self.root = root
        self.dataset = dataset
        self.type = type

        self.data_list = self.get_split_info()
        self.graph, self.text = self.parse_split()

        self.graph_data = [data[0] for data in self.data_list]
        self.text_data = [data[1] for data in self.data_list]

        self.graph_mol_data = MoleculeDataset(self.graph_data)
        self.data = [[text, graph] for text, graph in zip(self.text_data, list(self.graph_mol_data))]

        print(f'# {self.type} smiles: %d '% (len(self.data)))
    def get_split_info(self):

        with codecs.open(os.path.join(self.root, file), 'r',encoding='utf-8') as f:
            data = csv.DictReader(f, skipinitialspace=True, delimiter='\t')
            data_list = []
            for instance in data:
                if instance['smiles'] != '*':
                    smiles, description = instance['smiles'], instance['description']
                    description = description.replace('\n', '')
                else:
                    print("smiles error: ", instance['cid'])

                mol = get_mol(smiles)
                if mol is None:
                    continue
                else:
                    data_i = [smiles, description]
                    data_list.append(data_i)
        return data_list

    def parse_split(self):
        graph, text = zip(*self.data_list)
        return graph, text


    def __getitem__(self, index):
        text, smiles = self.data[index]
        data = [
            text, smiles, index
        ]
        return data

    def __len__(self):
        return len(self.data)

