# PEMDAS_B351Proj
 A project for Spring 2023 at IU Bloomington for B351 - Intro to Artificial Intelligence

## Team Members
- Praneeth Bhattiprolu
- Trevor Buechler
- Thomas Pugh
- Cullen Sullivan
- Zeshawn Zahid

## Project Description
Using spectra data and machine learning to predict whether a molecule contains nitrogen or not. Code in this repository includes code for data harvesting, NMR simulation, and machine learning models.

## File Structure
- 'data_harvesting' - contains the code for the data harvesting
- 'nmr_sim' - contains the code for the NMR simulation
- 'spectra' - contains the NMR spectra data in a jdx format
- 'iris.data' - contains the data for the iris dataset used for practicing machine learning while waiting on the NMR data
- 'ml_libraries.py' - contains the code using third party libraries for machine learning
- 'scratch_mlp.py' - contains the code for the scratch implementation of the MLP. This was scrapped in favor of the third party libraries
- Spectra.ipynb - notebook that contains the code and results for the machine learning models
- 'Nitrogenic.zip' - contains the NMR spectra data for molecules that contain nitrogen
- 'NoNitrogenic.zip' - contains the NMR spectra data for molecules that do not contain nitrogen
- test_ir_consistency.py - contains the code for testing the consistency of the IR spectra data

## How to Run
- Data harvesting code does not need to be run as the data is already in the repository in the zip files
- NMR simulation code does not need to be run as the data is already in the repository in the zip files
- To run, upload the Spectra.ipynb file to Google Colab
- In Google Colab, upload data_harvesting/spectra_reader.py, Nitrogenic.zip, and NoNitrogenic.zip
- You can then run all the cells in the notebook