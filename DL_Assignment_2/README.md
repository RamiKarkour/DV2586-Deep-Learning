# Assignment 2: LSTM Autoencoder for Anomaly Detection

This repository contains the code and report for Assignment 2 in the Deep Learning course.  
The project uses an LSTM autoencoder approach for anomaly detection on a Network Intrusion Detection dataset.

## Dataset

The selected dataset is **NSL-KDD**, used here as a Network Intrusion Detection dataset.  
The task is converted into a binary anomaly detection problem:

- `normal` traffic = normal behavior
- all attack categories = anomaly

The dataset is downloaded/loaded by the notebook and should not be uploaded to GitHub.

## Project Structure

```text
assignment2/
├── README.md
├── .gitignore
├── report.pdf
├── assignment2_lstm_anomaly_detection_final.ipynb
├── assignment2_requirements_final.txt
└── outputs/
    ├── metrics/
    │   ├── data_summary.json
    │   ├── evaluation_results.csv
    │   ├── evaluation_results.json
    │   ├── lstm_ae_small_history.json
    │   └── lstm_ae_deep_history.json
    └── plots/
        ├── lstm_ae_small_training_history.png
        ├── lstm_ae_deep_training_history.png
        ├── lstm_ae_small_reconstruction_error.png
        ├── lstm_ae_deep_reconstruction_error.png
        ├── lstm_ae_small_error_histogram.png
        ├── lstm_ae_deep_error_histogram.png
        ├── lstm_ae_small_confusion_matrix.png
        ├── lstm_ae_deep_confusion_matrix.png
        ├── isolation_forest_anomaly_scores.png
        ├── isolation_forest_score_histogram.png
        ├── isolation_forest_confusion_matrix.png
        └── model_comparison.png
```

## Models

The notebook trains and evaluates:

1. A smaller LSTM autoencoder
2. A deeper LSTM autoencoder
3. Isolation Forest as a baseline model

The LSTM autoencoders are trained only on normal traffic sequences.  
A separate tuning split is used to select anomaly thresholds, and the final test split is used for final evaluation.

## Metrics

The models are compared using:

- Precision
- Recall
- F1-score
- ROC-AUC
- AUC-PR
- Confusion matrix

## How to Run

Create or activate a Python environment with the required packages:

```bash
pip install -r assignment2_requirements_final.txt
```

Then open the notebook:

```bash
jupyter lab
```

Run:

```text
assignment2_lstm_anomaly_detection_final.ipynb
```

The notebook saves plots and metrics automatically inside the `outputs/` folder.

## Notes

The dataset folder, cached files, checkpoints, and notebook checkpoint folders are intentionally excluded from GitHub.  
The report explains the preprocessing, model design, anomaly thresholding, evaluation results, and limitations.
