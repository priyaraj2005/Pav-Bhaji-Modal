# DriveBuddyAI Pav-Bhaji Text Classification

## Objective

Classify whether an Instagram post image is Pav-Bhaji or not using
Instagram metadata and text features.

This solution follows the challenge requirement to use a text-based
classification approach rather than CNN/image-pixel classification.

## Dataset

452 labeled images were successfully matched with metadata.

Class distribution:

- Non-Pav-Bhaji: 269
- Pav-Bhaji: 183

## Features

The model uses:

- Instagram captions
- Instagram tags and hashtags
- Word-level TF-IDF features
- Character-level TF-IDF features

Direct Pav-Bhaji terms were removed to reduce target leakage.

## Model

The final model uses:

- Word TF-IDF: 1-2 grams
- Character TF-IDF: 3-5 grams
- LinearSVC
- C = 0.1
- Balanced class weights
- Random state = 42

## Evaluation

Stratified 80/20 split:

- Training samples: 361
- Test samples: 91

Held-out test results:

- Accuracy: 60.44%
- Pav-Bhaji F1: 61.70%
- ROC-AUC: 67.02%

Confusion matrix:

[[26, 28],
 [8, 29]]

## Final Model

After evaluation, the selected model was retrained using all 452
matched labeled samples.

The trained model is saved as:

priya_raj_drivebuddyai_model.joblib

## Files

- priya_raj.py
- priya_raj_data_analysis_report.pdf
- priya_raj_drivebuddyai_model.joblib
- priya_raj_predictions.csv
- README.md
- requirements.txt

## Installation

pip install -r requirements.txt

## Notes

This project uses Instagram metadata/text rather than image pixels,
as required by the challenge.
