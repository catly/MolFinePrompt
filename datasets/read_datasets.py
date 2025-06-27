import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
print(DIR_PATH)
DATASET_PATHS = {
    "drugbank": os.path.join(DIR_PATH, "../data/drugbank"),
    "pubchem": os.path.join(DIR_PATH, "../data/pubchem"),
    "drugbank_DDI" : os.path.join(DIR_PATH, "../data/ddi/drugbank_DDI"),

    "kv_data" : os.path.join(DIR_PATH, "../data/kv_data"),
    "phy_data" : os.path.join(DIR_PATH, "../data/kv_data/phy_data"),
    "pubchem_test": os.path.join(DIR_PATH, "../data/pubchem"),
    "ChEBI-20_data" : os.path.join(DIR_PATH, "../data/ChEBI-20_data"),

    "MoleculeNet": os.path.join(DIR_PATH, "../data/MoleculeNet"),

    "DDI": os.path.join(DIR_PATH, "../data/ddi"),
}