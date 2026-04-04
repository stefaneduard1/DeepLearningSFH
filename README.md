# DeepLearningSFH

## Overview

This project explores the use of deep learning models to predict stellar population properties from synthetic spectral data.

Given a galaxy spectrum, the goal is to predict key physical parameters:

* Log average age (`logage_in`)
* Metallicity (`metal_in`)
* Dust extinction (`ebv_in`)
* Mass-to-light ratio (`ML_r`)

The project focuses on comparing different modelling strategies, including:

* Baseline convolutional neural networks (CNNs)
* Deeper CNN architectures
* Training on specific stellar population subsets
* Training with different loss functions

---

## Core Idea

All scripts in this repository follow a **common pipeline**, applied to different experimental setups:

```
Load data → Run model → Compare to true values → Plot results
```

Rather than being completely separate pieces of code, the scripts represent **variations of the same workflow**, used to investigate how different modelling choices affect performance. These codes were changed constantly and used at different times.

---

## Experiments

The project consists of several experiments built on this shared pipeline:

### Baseline Model

* Simple CNN architecture
* Trained using mean squared error (MSE)
* Serves as a reference for comparison

---

### Deeper Model

* Increased model capacity (more layers / parameters)
* Aims to improve performance on complex spectral features

---

### Seperated Training (Stellar Populations)

* Data is filtered using the `fyoung` parameter
* Models are trained and evaluated on specific subsets:

  * Old populations (quiescent)
  * Intermediate populations (green valley)
  * Young populations (star forming)

This allows analysis of how model performance depends on the underlying stellar population.

---

### Loss Function Model

* Predicts both:
  * Mean values (μ)
  * Uncertainties (σ)
* Uses a custom probabilistic loss function
* Allows weighting of specific parameters (e.g. metallicity)

---

## Data

The dataset consists of synthetic stellar spectra stored in FITS files.

Each spectrum contains:

* `spec`: flux values
* `var`: variance (used to compute noise)

Labels are provided in a FITS table:

* `logage_in`
* `metal_in`
* `ebv_in`
* `ML_r`
* `fyoung` (used for binning experiments)

---

## Input Representation

Each sample is represented as:

```
(spectrum, noise) → shape (N_pixels, 2)
```

---

## Model Architecture

All models are based on convolutional neural networks:

* Convolutional layers extract spectral features
* Pooling reduces dimensionality
* Dense layers map features to physical parameters

---

## Evaluation

Each experiment produces a consistent set of diagnostic plots:

* **Predicted vs True**

  * Measures accuracy and bias

* **Residual distributions**

  * Shows error spread and systematic offsets

* **(For uncertainty models) Predicted σ distributions**

  * Evaluates learned uncertainties

---

## Outputs

The scripts generate:

* Trained models (`.keras`)
* Training histories (`.json`)
* Diagnostic plots:

  * Loss curves
  * Prediction accuracy plots
  * Residual distributions
  * Uncertainty distributions


