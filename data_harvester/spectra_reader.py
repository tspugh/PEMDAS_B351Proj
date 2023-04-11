import os
import jcamp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from numpy import zeros, ndarray

# PATH = "../spectra"
# OUTPUT_DIR = "../spectra/csv"
#
# TYPE_OPTIONS = {"MS", "IR", "UV", "1HNMR", "13CNMR"}
#
# # to be specified
# ARRAY_LENGTH_BY_TYPE = {
#     "MS": 0,
#     "IR": 0,
#     "UV": 0,
#     "1HNMR": 0,
#     "13CNMR": 0
# }
#
# # to be determined
# CLASSIFICATION_TO_VALUE = {
#     "amine": 0,
#     "2-amine": 1,
#     "3-amine": 2,
#     "nitrile": 3,
#     "imine": 4,
#     "amide": 5,  # double bond and single bond to Carbons
#     "nitro": 6,
#     "aziridine": 7,
#     "azetidine": 8,
#     "pyrrolidine": 9,
#     "piperidine": 10,
#     "azepane": 11,
#     "azocane": 12
# }
#
#
# class MoleculeObject:
#     def __init__(self, name: str, smiles: str, classification: str):
#         self.name = name
#         self.smiles = smiles
#         self.classification = classification
#         self.data_arrays = {}
#
#     def set_spectra_to_array(self, file, type):
#         if type in ["MS", "IR", "UV"]:
#             self.data_arrays.setdefault(type, jdx_to_array(file, type))
#         else:
#             self.data_arrays.setdefault(type, csv_to_array(file, type))
#
#     # TODO
#     def jdx_to_array(self, file, type):
#         array = list(zeros(ARRAY_LENGTH_BY_TYPE[type]))
#
#         # use jcamp!
#         # https://pypi.org/project/jcamp/
#
#         return None
#
#     # TODO
#     def csv_to_array(self, file, type):
#         array = list(zeros(ARRAY_LENGTH_BY_TYPE[type]))
#
#         # tbd
#
#         return None


class Molecule:
    def __init__(self, ir_filename=None, uv_filename=None, cnmr_filename=None, hnmr_filename=None, ms_filename=None,
                 class_filename=None):

        self.ir_data = None
        self.uv_data = None
        self.cnmr_data = None
        self.hnmr_data = None
        self.ms_data = None
        self.classification_data = None

        # Store
        self.ms_filename = ms_filename
        self.hnmr_filename = hnmr_filename
        self.cnmr_filename = cnmr_filename

        # Initialization
        if ms_filename is not None:
            self.ms_data = self.read_ms_data()
        if hnmr_filename is not None:
            self.hnmr_data = self.read_csv_data('HNMR')
        if cnmr_filename is not None:
            self.cnmr_data = self.read_csv_data("CNMR")
        # if ir_filename is not None:
        #     self.ir_data = self.read(ir_filename)
        # if uv_filename is not None:
        #     self.uv_data = self.read(uv_filename)
        # if class_filename is not None:
        #     self.classification_data = self.read(class_filename)

    def read_ms_data(self):
        """ Reads the MS data, standardizes the array to set length of 200 and return the array """

        # read and process to numpy arrays using jcamp
        jcamp_dict = jcamp.jcamp_readfile(self.ms_filename)
        original_x_values = np.array(jcamp_dict['x'])
        original_y_values = np.array(jcamp_dict['y'])

        # max_length setter, can use tommys global dict later for readability
        max_length = 200

        # initialize
        y_values: ndarray = np.zeros(max_length)

        # fill the zero'd out y array comparing to X
        for i, x in enumerate(original_x_values):
            index = int(round(x))  # to compare, eh im being safe with the rounding and casting
            if index < max_length:  # just a check in case we are over 200, as well, though it seems they never get that high
                y_values[index] = original_y_values[i]  # and slot it in, x to correspond y in order

        return y_values

    def read_csv_data(self, data_type):
        """ Reads HNMR/CNMR data, fills in missing 0's between peaks and 0's before and after where data begins"""
        if data_type == 'HNMR':
            start = -1.99272
            stop = 10
            increment = 0.000336
            filename = self.hnmr_filename
        elif data_type == 'CNMR':
            start = -19.9524
            stop = 230
            increment = 0.001907
            filename = self.cnmr_filename

        data = pd.read_csv(filename, header=None)

        x_values = data[0].values[::-1]  # We need to reverse cuz it's backwards
        y_values = data[1].values[::-1]

        # Set start stop increments depending on HNMR or CNMR, and filename specs
        new_x_values = np.arange(start, stop, increment)
        new_y_values = np.zeros_like(new_x_values)

        for i, x_val in enumerate(x_values):
            index = int(round((x_val - start) / increment))  # get index that will be channeled into Y / other options are np.where(np.isclose) but I cant find good tolerances
            if 0 <= index < len(new_y_values):  # max length check
                new_y_values[index] = y_values[i]  # and slot it in there

        return new_y_values


test_m = Molecule(
    ms_filename='/Users/zesha/OneDrive/Desktop/School/IU/Spring 2023/AI/....FINAL_PROJECT/MS JDX/C924469_MS_0.jdx',
    hnmr_filename='/Users/zesha/OneDrive/Desktop/School/IU/Spring 2023/AI/....FINAL_PROJECT/PEMDAS_B351Proj/Nitrogenic/1-Butanesulfonamide/1H.csv')


# HNMR TEST
start = -1.99272
stop = 10
increment = 0.000336

hnmr_x_values = np.arange(start, stop, increment)
hnmr_y_values = test_m.hnmr_data
plt.plot(hnmr_x_values, hnmr_y_values) # **REMOVE*** this is for debugging
plt.show()


# MS TEST
# x_values = np.arange(len(test_m.ms_data))  # ***REMOVE** This is for debugging
# plt.figure()
# plt.stem(x_values, test_m.ms_data, 'm-', markerfmt=' ', basefmt=' ', linefmt='m-', use_line_collection=True)
# plt.show()
