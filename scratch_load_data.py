import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jcamp

''' Scratch / Testing area for CSV. Not implemented at all/just basic Ideas'''
# load via panda, data has no headers
data = pd.read_csv('2,4-Thiazolidinedione_1HNMR.csv', header=None)

# extract x and y values as separate arrays
x_values = data[0].values  # pd.read_csv()
y_values = data[1].values

# create numpy array of (x,y) points
nmr_data = np.column_stack((x_values, y_values))

# quick attempt at max length and padding the rest with 0's for now (FOR TRANSPOSED)
max_length = 5000
if nmr_data.shape[0] < max_length: # add zeros to end of array if data length is less than max length
    nmr_data = np.pad(nmr_data, ((0, max_length - nmr_data.shape[0]), (0, 0)), mode='constant')
elif nmr_data.shape[0] > max_length:  # truncate data if length is greater than max length
    nmr_data = nmr_data[:max_length]

# debug / see whats going
# np.set_printoptions(formatter={'all': lambda x: str(x) + ','})

print("NMR data shape:", nmr_data.shape)
print(nmr_data)

x_value = nmr_data[8, 0]
y_value = nmr_data[8, 1]

print(f'{x_value}\n{y_value}')

''' Scratch for JDX Reading of data. This is 'seemingly' (the core)  done but potential issues in consistency '''
filename = 'C628739_IR_0.jdx'
jcamp_dict = jcamp.jcamp_readfile(filename)  # load
jcamp.jcamp_calc_xsec(jcamp_dict, skip_nonquant=False, debug=True)  # calculate

max_length = 1000

y_values = np.zeros(max_length)  # trevor's initialize empty array with 0's of max length

# then fill the y_values array with the y values from the dictionary up to the maximum length
y_length = min(max_length, len(jcamp_dict['y']))
y_values[:y_length] = jcamp_dict['y'][:y_length]

x_values = list(range(len(y_values)))

plt.plot(x_values, y_values)
plt.show()

# SETTING UP OBJECT QUICK, Did this in like 15 minutes, the above is me debugging it, this is just the skeleton with
# quick copy and paste.
class SpectraData:
    def __init__(self, filepath_hnmr, filepath_cnmr):
        self.filepath_hnmr = filepath_hnmr
        self.filepath_cnmr = filepath_cnmr
        self.hnmr_data = None
        self.cnmr_data = None

        self.filepath_ir = filepath_ir
        self.ir_data = None

        # Read data on initialization..

    # Read CSV data

    def read_hnmr_data(self):
        """ CSV File, early implemntation, not setup. Will be more like JDX setup belo with max length, etc"""
        data = pd.read_csv(self.filepath_hnmr, header=None)
        x_values = data[0].values
        y_values = data[1].values
        self.hnmr_data = np.column_stack((x_values, y_values))

    def read_cnmr_data(self):
        """ CSV File, early implemntation, not setup. Will be more like JDX setup below with max length, etc"""
        data = pd.read_csv(self.filepath_cnmr, header=None)
        x_values = data[0].values
        y_values = data[1].values
        self.cnmr_data = np.column_stack((x_values, y_values))

    # Read JDX data
    def read_IR_data(self):
        """Jdx file, early implementation but has max length, possible consistency values."""
        jcamp_dict = jcamp.jcamp_readfile(self.filepath_ir)
        jcamp.jcamp_calc_xsec(jcamp_dict, skip_nonquant=False, debug=True)  # calculate

        max_length = 1000

        self.ir_data = np.zeros(max_length)  # trevor's initialize empty array with 0's of max length

        # then fill the y_values array with the y values from the dictionary up to the maximum length
        ir_data_length = min(max_length, len(jcamp_dict['y']))
        self.ir_data[:ir_data_length] = jcamp_dict['y'][:ir_data_length]

        # x_values = list(range(len(self.ir_data)))


    # def read_UV_data

    # def read_MS_data

    # Get Data
    def get_nmr_data(self):
        return self.hnmr_data

    def get_cnmr_data(self):
        return self.cnmr_data

    # Combine and refine data for classifier
