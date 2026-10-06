# Predicting Nitrogen from Molecular Spectra

This was a Spring 2023 project for Indiana University Bloomington's B351:
Introduction to Artificial Intelligence. It explores whether a molecule
contains nitrogen using combined IR, UV, mass-spectrometry, and NMR data.

## Team

- Praneeth Bhattiprolu
- Trevor Buechler
- Thomas Pugh
- Cullen Sullivan
- Zeshawn Zahid

## What is in the repository

- `Spectra.ipynb` contains the data-loading, preprocessing, model-training, and
  saved output from the final experiments.
- `data_harvester/` contains the spectra reader and collection utilities.
- `Nitrogenic.zip` and `NoNitrogen.zip` contain the two labelled spectra sets
  consumed by the notebook.
- `nmr_sim/` contains the NMR simulation work.
- `ml_libraries.py` contains earlier scikit-learn experiments.
- `scratch_MLP.py` is the team's earlier from-scratch MLP implementation; the
  final notebook uses library implementations instead.
- `iris.data` was used for ML practice while the spectra data was being
  assembled.
- `test_ir_consistency.py` contains exploratory IR consistency checks.

## Historical notebook result

The committed notebook records 272 usable molecule samples after loading the
archives. Its 100-neuron scikit-learn `MLPClassifier` output reports **85.45%
test accuracy** on the notebook's 80/20 split (47 correct predictions out of a
55-sample test set). The split does not set `random_state` explicitly, and the
notebook also shuffles the samples beforehand, so this is a historical saved
result rather than a claim that every new run will reproduce the same score.

## Run in Google Colab

Open `Spectra.ipynb` in Colab. Before running its existing cells, add and run a
setup cell that checks out the package code and places the two archives where
the notebook expects them:

```python
!git clone https://github.com/tspugh/PEMDAS_B351Proj.git /content/PEMDAS_B351Proj
!cp /content/PEMDAS_B351Proj/Nitrogenic.zip /content/Nitrogenic.zip
!cp /content/PEMDAS_B351Proj/NoNitrogen.zip /content/NoNitrogen.zip
!pip install jcamp

import sys
sys.path.insert(0, "/content/PEMDAS_B351Proj")
```

Then run the notebook cells in order. The archive cells extract `Nitrogenic/`
and `NoNitrogen/` under `/content`, and the import
`from data_harvester import spectra_reader` resolves from the cloned
repository. Colab already supplies the other libraries used by the notebook,
including NumPy, pandas, SciPy, scikit-learn, Matplotlib, and PyTorch.

The notebook contains saved output from the original coursework run, including
large intermediate arrays and model metrics. Re-running the training cells can
take time and may produce different metrics because the train/test split is not
fixed with an explicit seed.
