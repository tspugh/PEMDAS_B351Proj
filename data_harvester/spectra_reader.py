import os
import jcamp
from numpy import zeros

PATH = "../spectra"
OUTPUT_DIR = "../spectra/csv"

TYPE_OPTIONS = {"MS", "IR", "UV", "1HNMR", "13CNMR"}

#to be specified
ARRAY_LENGTH_BY_TYPE = {
    "MS": 0,
    "IR": 0,
    "UV": 0,
    "1HNMR": 0,
    "13CNMR": 0
}

#to be determined
CLASSIFICATION_TO_VALUE = {
    "amine": 0,
    "2-amine": 1,
    "3-amine": 2,
    "nitrile": 3,
    "imine": 4,
    "amide": 5, #double bond and single bond to Carbons
    "nitro": 6,
    "aziridine": 7,
    "azetidine": 8,
    "pyrrolidine": 9,
    "piperidine": 10,
    "azepane": 11,
    "azocane": 12
}

class MoleculeObject:
    def __init__(self, name:str, smiles:str,
                 classification:str):
        self.name = name
        self.smiles = smiles
        self.classification = classification
        self.data_arrays = {}

    def set_spectra_to_array(self, file, type):
        if(type in ["MS", "IR", "UV"]):
            self.data_arrays.setdefault(type, jdx_to_array(file, type))
        else:
            self.data_arrays.setdefault(type, csv_to_array(file,type))

    #TODO
    def jdx_to_array(self, file, type):
        array = list(zeros(ARRAY_LENGTH_BY_TYPE[type]))

        #use jcamp!
        # https://pypi.org/project/jcamp/

        return None

    #TODO
    def csv_to_array(self, file, type):
        array = list(zeros(ARRAY_LENGTH_BY_TYPE[type]))

        #tbd

        return None