# skin-lesion-analysis

This repository contains code to classify images of skin lesions to
detect melanoma.

## Data

Get data from kaggle:

`kaggle datasets download -d wanderdust/skin-lesion-analysis-toward-melanoma-detection`

Unzip to directory `skin-lesions/`

Training images are drawn uniformly from the three classes (Melanom, Nevus and
Seborrheic Keratosis), cropped to squares and rescaled to 224 by 224 RGB pixels.

## Model

The script `detect_melanom.jl` finetunes a pretrained ResNet (He et al. 2016)
with depth 50 to classify between Melanom, Nevus and Seborrheic Keratosis.
