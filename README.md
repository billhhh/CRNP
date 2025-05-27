# CRNP

The official code repository of ECCV 2022 paper "Uncertainty-aware Multi-modal Learning via Cross-modal Random Network Prediction"

## Installation

```commandline
pip install -r requirements.txt
```

For more requirements, please refer to requirements.txt

## Data Preparation

BraTS2020 dataset has 369 cases for training/validation and other 125 cases for evaluation, where each case (with four modalities, namely: Flair, T1, T1CE and T2) share one segmentation GT. Four classes (background included) are considered for each pixel.

The data can be requested from [here](https://ipp.cbica.upenn.edu/categories/brats2020) (3.3GB for training zip file and 1.2GB for validation zip file).

The data path can be changed in `datalist/BraTS20_train.csv` and `datalist/BraTS20_test.csv`.

## Model Training

Followed the official data splits of BraTS2020 settings, after the selection and hyper-parameters, the models are trained on training data for a certain iterations and then tested on public Validation data. For more hyper-parameters settings, please refer to `run.sh` and the paper.

For model training, the commandline is:

```commandline
bash run.sh [GPU id]
```

For instance:

```commandline
bash run.sh 0
```

## Model Evaluation

For model evaluation, the resume path of the tested model can be specified in the `eval.sh` file. The evaluation can be performed with:

```commandline
bash eval.sh [GPU id]
```

For example:

```commandline
bash eval.sh 0
```

Then perform postprocessing:

```commandline
python postprocess.py
```

The folder paths can be modified in `postprocess.py`.

After postprocessing, [online evaluation](https://ipp.cbica.upenn.edu/#BraTS20eval_validationPhase) is performed. Output folder containing 125 segmentations is required to upload to the site for evaluation.

Enjoy!!
