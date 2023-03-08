import os
import jcamp

PATH = "../spectra"
OUTPUT_DIR = "../spectra/csv"

TYPE_OPTIONS = {"MS", "IR", "UV", "TZ"}

#TODO - convert from jdx to csv
def convert_file(file:str, output_directory:str):
    pass

#TODO - read every folder in spectra, look into each of its subfolders, and convert all files of type to csv

#type can be from TYPE_OPTIONS
def run_over_all_dirs(type: str):
    #enter spectra

    #for every directory "enumerated" in spectra

    #for every subdirectory "mol" in "enumerated"

    #for each file "jdx_file" in "mol"

    #if "jdx_file" has "type" in the name, convert the file
    pass


def get_spectrum(file: str):
    return Spectrum(file)

class Spectrum:

    def __init__(self, file):
        self.parse_file(file)

    # TODO - initialize from the .jdx file
    def parse_file(self, file):
        pass
