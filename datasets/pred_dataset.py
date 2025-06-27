import codecs
import csv
import os
import numpy as np
from graph.chemutils import get_mol
import torch
from torch.utils.data import Dataset
from typing import List
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from graph.data_utils import MoleculeDataset

def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}

    for ind, smiles in enumerate(dataset):
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets

def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


def graph_to_mol(data):
    smiles, label = zip(*data)
    graph_mol_data = MoleculeDataset(smiles)
    data_list = [[graph, lab, smile] for graph, lab, smile in zip(list(graph_mol_data), list(label), list(smiles))]
    return data_list

class predDataset(Dataset):
    def __init__(
            self,
            root,
            task,
            split,
            num_tasks
    ):
        self.root = root
        self.task = task
        self.split = split
        self.valid_size = 0.1
        self.test_size = 0.1
        self.num_tasks = num_tasks

        file = os.path.join(self.root, self.task, f'{self.task}.csv')
        self.smiles_data, self.data = self.read_preddata(file)

        self.train_data, self.valid_data, self.test_data = self.get_split()
        self.train_dataset, self.valid_dataset, self.test_dataset = self.get_dataset()
    
        print(f'# {self.task} train_data: %d  val_data: %d  text_data: %d ' % (
        len(self.train_dataset), len(self.valid_dataset), len(self.test_dataset)))

    def read_preddata(self, file):
        smiles_column_name = "smiles"
        if self.task == 'bbbp':
            label_column_name = ["p_np"]
        elif self.task == "bace":
            label_column_name = ["Class"]
        elif self.task == "clintox":
            label_column_name = ["FDA_APPROVED","CT_TOX"]
        elif self.task == "tox21":
            label_column_name = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
       'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        elif self.task == "sider":
            label_column_name = ['Hepatobiliary disorders',
             'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
             'Investigations', 'Musculoskeletal and connective tissue disorders',
             'Gastrointestinal disorders', 'Social circumstances',
             'Immune system disorders', 'Reproductive system and breast disorders',
             'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
             'General disorders and administration site conditions',
             'Endocrine disorders', 'Surgical and medical procedures',
             'Vascular disorders', 'Blood and lymphatic system disorders',
             'Skin and subcutaneous tissue disorders',
             'Congenital, familial and genetic disorders',
             'Infections and infestations',
             'Respiratory, thoracic and mediastinal disorders',
             'Psychiatric disorders', 'Renal and urinary disorders',
             'Pregnancy, puerperium and perinatal conditions',
             'Ear and labyrinth disorders', 'Cardiac disorders',
             'Nervous system disorders',
             'Injury, poisoning and procedural complications']
        elif self.task == "muv":
            label_column_name = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
             'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
             'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']
        elif self.task == "hiv":
            label_column_name = ['HIV_active']

        smiles_label_list = []
        smiles_list = []
        with codecs.open(file, 'r',encoding='utf-8') as f:
            data = csv.DictReader(f, skipinitialspace=True, delimiter=',')
            headers = data.fieldnames

            for instance in data:
                label = []
                smiles = instance[smiles_column_name]
                if self.task == "toxcast":
                    label_column_name = headers[1:]
                if self.task == "pcba":
                    label_column_name = headers[:-2]
                for lab_num in label_column_name:
                    lab = instance[lab_num]
                    if lab != '0' and lab != '1' and lab != '0.0' and lab != '1.0':
                        lab = 0.5
                    label.append(float(lab))
                mol = get_mol(smiles)
                if mol is None:
                    continue
                else:
                    row_data = [smiles, label]
                    smiles_list.append(smiles)
                    smiles_label_list.append(row_data)
        return smiles_list, smiles_label_list

    def get_split(self):
        if self.split == 'random':
            num_train = len(self.smiles_data)
            indices = list(range(num_train))
            np.random.shuffle(indices)

            split = int(np.floor(self.valid_size * num_train))
            split2 = int(np.floor(self.test_size * num_train))
            valid_idx, test_idx, train_idx = indices[:split], indices[split:split + split2], indices[split + split2:]
        elif self.split == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(self.smiles_data, self.valid_size, self.test_size)

        train_data = [self.data[idx] for idx in train_idx]
        valid_data = [self.data[idx] for idx in valid_idx]
        test_data = [self.data[idx] for idx in test_idx]

        return train_data, valid_data, test_data

    def get_dataset(self):
        train_dataset = graph_to_mol(self.train_data)
        valid_dataset = graph_to_mol(self.valid_data)
        test_dataset = graph_to_mol(self.test_data)
        return train_dataset, valid_dataset, test_dataset

    def __len__(self):
        return len(self.data)