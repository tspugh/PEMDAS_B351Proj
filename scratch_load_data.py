import numpy as np
import pandas as pd

# load via panda, data has no headers
data = pd.read_csv('2,4-Thiazolidinedione_1HNMR.csv', header=None)
# data = np.genfromtxt('2,4-Thiazolidinedione_1HNMR.csv', delimiter=',')

# extract x and y values as separate arrays
x_values = data[0].values  # pd.read_csv()
y_values = data[1].values

# create numpy array of (x,y) points
nmr_data = np.column_stack((x_values, y_values))  # Not transposed
# nmr_data = np.column_stack((x_values, y_values)).T  # or Transposed, depending on what's better for the classifier.

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

# x_value = nmr_data[0, 8]  # transposed, line 9 in csv test
# y_value = nmr_data[1, 8]
print(f'{x_value}\n{y_value}')


# SETTING UP OBJECT QUICK, Did this in like 15 minutes, the above is me debugging it, this is just the skeleton with
# quick copy and paste.
class SpectraData:
    def __init__(self, filepath_hnmr, filepath_cnmr):
        self.filepath_hnmr = filepath_hnmr
        self.filepath_cnmr = filepath_cnmr
        self.hnmr_data = None
        self.cnmr_data = None

    def read_hnmr_data(self):
        data = pd.read_csv(self.filepath_hnmr, header=None)
        x_values = data[0].values
        y_values = data[1].values
        self.hnmr_data = np.column_stack((x_values, y_values)).T

    def read_cnmr_data(self):
        data = pd.read_csv(self.filepath_cnmr, header=None)
        x_values = data[0].values
        y_values = data[1].values
        self.cnmr_data = np.column_stack((x_values, y_values)).T

    # read_other_data

    def get_nmr_data(self):
        return self.hnmr_data

    def get_cnmr_data(self):
        return self.cnmr_data

    # get_other_data

    # combine and refine data for classifier
