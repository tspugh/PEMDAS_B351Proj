import os
import jcamp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

class IRError(Exception):
    pass

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

    IR_LOWER_REQ = 600
    IR_UPPER_REQ = 3800

    def __init__(self, ir_filename=None, uv_filename=None, cnmr_filename=None, hnmr_filename=None, ms_filename=None,
                 class_filename=None, debug=False):

        self.ir_data = None
        self.uv_data = None
        self.cnmr_data = None
        self.hnmr_data = None
        self.ms_data = None
        self.classification_data = None

        self.debug = debug

        self.invalid_flag = False

        # Store
        self.ms_filename = ms_filename
        self.hnmr_filename = hnmr_filename
        self.cnmr_filename = cnmr_filename
        self.ir_filename = ir_filename

        # Initialization
        if ms_filename is not None:
            self.ms_data = self.read_ms_data()
        if hnmr_filename is not None:
            self.hnmr_data = self.read_csv_data('HNMR')
        if cnmr_filename is not None:
            self.cnmr_data = self.read_csv_data("CNMR")
        if ir_filename is not None:
            self.ir_data = self.read_ir_data(ir_filename)
        # if uv_filename is not None:
        #     self.uv_data = self.read(uv_filename)
        # if class_filename is not None:
        #     self.classification_data = self.read(class_filename)

        # THE ARRAY
        self.monster_array = self.combine_data()

    def read_ms_data(self):
        """ Reads the MS data, standardizes the array to set length of 200 and return the array """
        # read and process to numpy arrays using jcamp
        jcamp_dict = jcamp.jcamp_readfile(self.ms_filename)
        original_x_values = np.array(jcamp_dict['x'])
        original_y_values = np.array(jcamp_dict['y'])

        # max_length setter, can use tommys global dict later for readability
        max_length = 200

        # initialize
        y_values = np.zeros(max_length)

        # fill the zero'd out y array comparing to X
        for i, x in enumerate(original_x_values):
            index = int(round(x))  # to compare, eh im being safe with the rounding and casting
            if index < max_length:  # just a check in case we are over 200, as well, though it seems they never get that high
                y_values[index] = original_y_values[i]  # and slot it in, x to correspond y in order

        if self.debug: print("Successful MS Read")

        return y_values

    def read_csv_data(self, data_type):
        """ Reads HNMR/CNMR data, fills in missing 0's between peaks and 0's before and after where data begins"""
        if data_type == 'HNMR':
            start = -1.99272
            stop = 10
            increment = 0.000336
            filename = self.hnmr_filename
        else:
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

        if self.debug: print(f'Successful CSV read. Shape: {new_y_values.shape}')

        return new_y_values

    def read_ir_data(self):
        try:
            jcamp_dict = jcamp.jcamp_readfile(self.ir_filename)
            x = jcamp_dict["x"]
            y = jcamp_dict["y"]

            fit_length = 56420

            if jcamp_dict["xunits"].lower() == "micrometers":
                raise IRError("Data Unit Invalid")
            if x[0] > self.IR_LOWER_REQ or x[-1] < self.IR_UPPER_REQ:
                raise IRError("Data Range Invalid")

            if x[0] > x[1]:
                x = x[::-1]
                y = y[::-1]

            new_x = np.zeros(fit_length)
            new_y = np.zeros(fit_length)

            min_index = 0
            max_index = len(x)-1
            while x[min_index] < self.IR_LOWER_REQ:
                min_index+=1
            while x[max_index] > self.IR_UPPER_REQ:
                max_index-=1

            index_range = max_index-min_index
            ratio = (fit_length-1)/index_range

            new_x[fit_length-1] = x[max_index]
            new_y[fit_length-1] = y[max_index-1]

            index = 0
            #interpolation
            for k in range(fit_length-1):
                if 0 <= k%ratio < 1:
                    new_x[k] = x[index]
                    new_y[k] = y[index]
                    index += 1
                else:
                    new_x[k] = (x[index]-x[index-1])*k/ratio + x[
                        index-1]
                    new_y[k] = (y[index] - y[index - 1]) * k / ratio + \
                               y[index - 1]

            return np.concatenate(new_x, new_y)

        except IRError as er:
            if self.debug: print(f"IR Data Incompatible: {er}")
            self.invalid_flag = True
        except Exception as e:
            if self.debug: print(e)
            self.invalid_flag = True

        return None


    def combine_data(self):
        """combine the data in order: starting index will be 1 or 0 depnding if it has nitrogen (handled in loading data)
        then cnmr_data.shape: 131072,
        then hnmr_data: 35693,
        then ms_data: 200) """
        self.monster_array = np.concatenate((self.cnmr_data, self.hnmr_data, self.ms_data))
        return self.monster_array


def load_data_both(debug=False):
    # When used on its own set the directory to .. otherwise leave it empty as we will use it as an import
    root_directory = ''

    # define the filenames to search for
    filenames_to_search = ['*_IR_*.jdx', '*_UV_*.jdx', '13C.csv', '1H.csv', '*_MS_*.jdx', 'classification_info.txt']

    # create an empty list to hold the Molecule objects
    molecules = []

    # iterate over each molecule directory in the Nitrogenic and NoNitrogen directories
    for dir_name in ['Nitrogenic', 'NoNitrogen']:
        subdirectory = os.path.join(root_directory, dir_name)
        for molecule_dir in os.listdir(subdirectory):
            # create a dictionary to hold the filenames for this molecule
            filetype_to_filenames = {}

            # iterate over each filename pattern to search for
            for pattern in filenames_to_search:
                # create the full search pattern for the current file type
                search_pattern = os.path.join(subdirectory, molecule_dir, pattern)

                # search for all files matching the pattern and add them to the list for the current file type
                matched_filenames = glob.glob(search_pattern)
                filetype_to_filenames[pattern] = matched_filenames

            # try to create the Molecule object with the filenames
            try:
                ir_filename = next(iter(filetype_to_filenames['*_IR_*.jdx']), None)
                uv_filename = next(iter(filetype_to_filenames['*_UV_*.jdx']), None)
                cnmr_filename = next(iter(filetype_to_filenames['13C.csv']), None)
                hnmr_filename = next(iter(filetype_to_filenames['1H.csv']), None)
                ms_filename = next(iter(filetype_to_filenames['*_MS_*.jdx']), None)
                class_filename = next(iter(filetype_to_filenames['classification_info.txt']), None)

                if debug:
                    molecule = Molecule(ir_filename, uv_filename, cnmr_filename, hnmr_filename, ms_filename,
                                        class_filename, debug=True)
                else:
                    molecule = Molecule(ir_filename, uv_filename, cnmr_filename, hnmr_filename, ms_filename,
                                        class_filename)

                if dir_name == 'Nitrogenic':
                    molecule.monster_array = np.insert(molecule.monster_array, 0, 1)  # 1 for Nitrogenic
                else:
                    molecule.monster_array = np.insert(molecule.monster_array, 0, 0)  # 0 for Not
                if not molecule.invalid_flag:
                    molecules.append(molecule)
            except Exception as e:
                print(f"Error processing files in '{os.path.join(subdirectory, molecule_dir)}': {e}")
                continue

    return molecules


if __name__ == "__main__":
    molecule_data = load_data_both(debug=False)

    # MS TEST
    # x_values = np.arange(len(molecule_data[2].ms_data))  # ***REMOVE** This is for debugging
    # plt.figure()
    # plt.stem(x_values, molecule_data[2].ms_data, 'm-', markerfmt=' ', basefmt=' ', linefmt='m-',
    #          use_line_collection=True)
    # plt.show()

    # Get MonsteR ArrAY

    print(f'Monster array size. First index is 0 or 1 for having nitrogenic group (1 is true) {molecule_data[0].monster_array.shape}')
    print(f"After that is CNMR data. Array size {molecule_data[0].cnmr_data.shape}")
    print(f"After that is HNMR data. Array size {molecule_data[0].hnmr_data.shape}")
    print(f"After that is MS data. Array size {molecule_data[0].ms_data.shape}")
    print(
        f"After that is IR data. Array size"
        f" {molecule_data[0].ir_data.shape}")
    print(f'Folders loaded {len(molecule_data)}') # Loading 341/352


    ##### AND THE FINISH #### didnt test this
    monster_arrays = [mol.monster_array for mol in molecule_data]
    data_table = np.vstack(monster_arrays)

