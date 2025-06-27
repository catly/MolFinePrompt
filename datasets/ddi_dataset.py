import codecs
import csv
import os
import pickle
from graph.chemutils import get_mol
import torch
from torch.utils.data import Dataset
from graph.data_utils import MoleculeDataset

def graph_to_mol(data):
    smiles1, smiles2, label = zip(*data)
    graph_mol_data1 = MoleculeDataset(smiles1)
    graph_mol_data2 = MoleculeDataset(smiles2)
    data_list = [[smile1, smile2, lab] for smile1, smile2, lab in zip(list(graph_mol_data1), list(graph_mol_data2), list(label))]
    return data_list

class ddiDataset(Dataset):
    def __init__(
            self,
            root,
            task,
            num_tasks
    ):
        self.root = root
        self.task = task
        self.num_tasks = num_tasks
        self.split_mode = True
        self.data_type = False

        self.train_data = self.read_data("train")
        self.valid_data = self.read_data("valid")
        self.test_data = self.read_data("test")

        self.train_dataset, self.valid_dataset, self.test_dataset = self.get_dataset()

        print(f'# {self.task} train_data: %d  val_data: %d  text_data: %d ' % (
        len(self.train_dataset), len(self.valid_dataset), len(self.test_dataset)))

    def read_data(self, type):
        smiles1_list, smiles2_list, smiles_label_list = [], [], []
        with codecs.open(os.path.join(self.root, self.task, f'{type}.csv'), 'r',encoding='utf-8') as f:
            data = csv.DictReader(f, skipinitialspace=True, delimiter=',')
            for instance in data:
                smiles1 = instance["smiles_1"]
                smiles2 = instance["smiles_2"]
                if self.task == 'DrugbankDDI':
                    label = int(instance["type"])
                else:
                    label = instance["label"]

                mol1 = get_mol(smiles1)
                mol2 = get_mol(smiles2)
                if mol1 is None or mol2 is None:
                    continue
                else:
                    row_data = [smiles1, smiles2, int(label)]
                    smiles_label_list.append(row_data)
        return smiles_label_list

    def get_dataset(self):
        train_dataset = graph_to_mol(self.train_data)
        valid_dataset = graph_to_mol(self.valid_data)
        test_dataset = graph_to_mol(self.test_data)
        return train_dataset, valid_dataset, test_dataset

    def __len__(self):
        return len(self.data)