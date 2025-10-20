# Bird Species Classifier

This repository contains code, notebooks, and utilities for training and evaluating deep-learning models to classify bird species from images.

The project collects model experiments (Keras/TensorFlow) and helper scripts used to prepare folders, train models, and test results.

## Repository structure

- `app.py` — (optional) top-level script / entrypoint. Check the file to see how it's used in this project.
- `requirements.txt` — Python dependencies used by notebooks and scripts.
- `Models/` — Jupyter notebooks that contain model training and experimentation. Notable notebooks:
	- `InceptionV3.ipynb` — training/transfer-learning experiment with InceptionV3.
	- `MobileNet.ipynb` — MobileNet-based experiment.
	- `VGG16 birds-classification-using-tflearning.ipynb` — VGG16 transfer learning for bird classification.
	- `VGG19.ipynb` — VGG19-based experiment.
	- `Xception.ipynb` — Xception-based experiment.
- `Utils/` — helper scripts and utilities used across notebooks and experiments:
	- `All_folders.py` — utilities for preparing dataset folders and paths.
	- `Bird_Classification.py` — helper functions for dataset loading, preprocessing, and model utilities.
	- `folders_name.txt` — (likely) list of folder/dataset names used by scripts.
	- `model_testing.ipynb` — notebook for testing trained models and evaluating metrics.

## Goal

The main goal of this repo is to provide an experimental playground for training image classification models (transfer learning and custom training) to recognize bird species. Notebooks should contain the training loops, data generators, evaluation metrics, and model saving/loading logic.

## Quick setup

1. Create and activate a Python virtual environment (recommended):

	 ```
     python3 -m venv .venv
	 source .venv/bin/activate
     ```

2. Install dependencies:

	 pip install -r requirements.txt

3. Inspect `Utils/All_folders.py` and `Utils/Bird_Classification.py` to verify expected dataset layout and any paths that must be adjusted.

4. Put your dataset in the layout expected by the utilities (usually a root folder with one subfolder per class). See `Utils/folders_name.txt` for expected class names if provided.

5. Run a notebook from the `Models/` directory using Jupyter or JupyterLab to train or evaluate a model. Example:

	 jupyter lab Models/InceptionV3.ipynb

Or run `app.py` if it is implemented as a CLI/entrypoint (open the file to see available options).

## Notebooks

- Each notebook in `Models/` is an experiment. They typically include sections for:
	- dataset preparation and augmentation
	- model definition (transfer learning from a backbone)
	- training and checkpointing
	- evaluation and visualization (confusion matrix, sample predictions)

Open the notebooks in order to reproduce or adapt experiments. `model_testing.ipynb` (under `Utils/`) contains code to load saved models and run tests on held-out images.

## Utilities

- `Utils/All_folders.py` — provides helper functions to create dataset splits or to reorganize images into train/val/test folders.
- `Utils/Bird_Classification.py` — likely contains dataset loaders, preprocessing functions, and convenience wrappers for Keras ImageDataGenerator or tf.data pipelines.

Before running the notebooks, review these scripts and update paths or parameters specific to your local dataset.