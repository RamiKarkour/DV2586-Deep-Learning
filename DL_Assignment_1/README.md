# Deep Learning Assignment 1

This repository contains the implementation and report for Deep Learning Assignment 1. The work includes gradient descent optimization and image classification on CIFAR-100 using a custom CNN and several established convolutional neural network architectures.

## Contents

```text
.
├── report.pdf
├── q1_gradient_descent.py
├── q2_train_models.py
├── q2_evaluation.py
├── requirements.txt
└── outputs/
    ├── filters/
    │   ├── custom_cnn_filters_before.png
    │   └── custom_cnn_filters_after.png
    ├── metrics/
    │   ├── evaluation_results.csv
    │   ├── evaluation_results.json
    │   └── training_histories.json
    └── plots/
        ├── q2_training_curves_all_models.png
        ├── q2_accuracy_comparison.png
        ├── q2_f1_comparison.png
        └── q2_confusion_matrix_preview.png
```

## Question 1: Gradient Descent

The first part implements gradient descent from scratch for the function:

```text
f(x, y) = (x - 3)^2 + (y + 2)^2
```

The script calculates the partial derivatives, runs gradient descent, and saves the optimization plot.

Run:

```bash
python q1_gradient_descent.py
```

Expected output:

```text
q1_gradient_descent.png
```

## Question 2: CIFAR-100 Image Classification

The second part trains and evaluates five models on CIFAR-100:

- Custom CNN
- ResNet-50
- VGG-19 BN
- DenseNet-121
- EfficientNet-B0

The training script saves model histories, plots, metrics, and filter visualizations.

Run training:

```bash
python q2_train_models.py
```

Run evaluation:

```bash
python q2_evaluation.py
```

## Results Summary

| Model | Accuracy | Macro F1-score |
|---|---:|---:|
| Custom CNN | 55.99% | 0.5531 |
| ResNet-50 | 64.15% | 0.6379 |
| VGG-19 BN | 66.66% | 0.6642 |
| DenseNet-121 | 60.57% | 0.6018 |
| EfficientNet-B0 | 45.88% | 0.4467 |

## Reproducibility

Install the required packages:

```bash
pip install -r requirements.txt
```

The CIFAR-100 dataset is downloaded automatically by the code through `torchvision.datasets.CIFAR100`.

## Notes

The dataset and trained checkpoint files are not included in the repository. The code can be rerun to reproduce the outputs.
