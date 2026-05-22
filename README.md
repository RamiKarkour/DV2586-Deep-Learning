# DV2586 Deep Learning

This repository contains the coursework developed for the DV2586 Deep Learning course. It is organized into two assignments and one final project. Each folder contains the source code, report material, saved metrics, and relevant output plots for that specific task.

## Repository structure

```text
DV2586-Deep-Learning/
├── DL_assignment_1/
│   ├── DL_Assignment1_report.pdf
│   ├── q1_gradient_descent.py
│   ├── q2_train_models.py
│   ├── q2_evaluation.py
│   ├── requirements.txt
|   ├── README.md
│   └── outputs/
│       ├── filters/
│       ├── metrics/
│       └── plots/
│
├── assignment2/
│   ├── Assignment2_LSTM_Report.pdf
│   ├── assignment2_lstm_anomaly_detection_final.ipynb
│   ├── assignment2_requirements_final.txt
|   ├── README.md
│   └── outputs/
│       ├── metrics/
│       └── plots/
│
└── DL_Final_Project/
    ├── README.md
    ├── requirements.txt
    ├── notebooks/
    ├── data/
    │   └── processed/
    └── outputs/
        ├── metrics/
        ├── plots/
        └── results/
```

## Assignment 1: Gradient Descent and CIFAR-100 Image Classification

Assignment 1 contains two main parts. The first part implements gradient descent on a simple two-variable function and visualizes the optimization path. The second part trains and evaluates a custom CNN and several pretrained CNN architectures on CIFAR-100.

### Main tasks

- Implement gradient descent and show the loss curve and optimization trajectory.
- Train a custom CNN on CIFAR-100.
- Fine-tune pretrained models, including ResNet-50, VGG-19, DenseNet-121, and EfficientNet-B0.
- Compare the models using accuracy, F1-score, confusion matrix information, and training curves.
- Visualize the first-layer filters of the custom CNN before and after training.

### Main files

```text
assignment1/
├── report.pdf
├── q1_gradient_descent.py
├── q2_train_models.py
├── q2_evaluation.py
├── requirements.txt
└── outputs/
```

## Assignment 2: LSTM Autoencoders for Anomaly Detection

Assignment 2 focuses on anomaly detection using the NSL-KDD network intrusion detection dataset. The work compares two LSTM autoencoder models with an Isolation Forest baseline.

### Main tasks

- Load and preprocess the NSL-KDD intrusion detection dataset.
- Encode categorical features and scale numerical features.
- Create fixed-length sequences for LSTM-based reconstruction.
- Train two LSTM autoencoder variants on normal traffic behavior.
- Select anomaly thresholds using a tuning split.
- Evaluate the models on a final test split using precision, recall, F1-score, ROC-AUC, and AUC-PR.
- Compare the LSTM autoencoders against Isolation Forest.
- Save reconstruction-error plots, histograms, confusion matrices, metrics, and model-comparison plots.

### Main files

```text
assignment2/
├── report.pdf
├── assignment2_lstm_anomaly_detection_final.ipynb
├── assignment2_requirements_final.txt
└── outputs/
    ├── metrics/
    └── plots/
```

## Final Project: Resume and Job Matching

The final project explores resume-to-job matching using natural language processing and deep learning techniques. The project compares traditional information retrieval methods with sentence-transformer-based approaches and a hybrid ranking method.

### Main tasks

- Prepare resume and job posting data.
- Build and evaluate baseline ranking methods such as TF-IDF and BM25.
- Use sentence-transformer embeddings for semantic matching.
- Fine-tune an SBERT-based model for improved matching.
- Compare baseline, pretrained, fine-tuned, and hybrid methods.
- Analyze ranking performance using metrics such as Top-1 accuracy, Top-5 ranking behavior, and MRR.
- Save final metrics, result files, and plots for analysis.

### Main files and folders

```text
DL_Final_Project/
├── README.md
├── requirements.txt
├── notebooks/
│   └── final_project_resume_job_matching.ipynb
├── data/
│   └── processed/
└── outputs/
    ├── metrics/
    ├── plots/
    └── results/
```

## Environment

The projects were developed using Python with common deep learning and data science libraries, including:

- PyTorch
- TorchVision
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook / JupyterLab
- Sentence Transformers, for the final project

Each assignment or project folder contains its own requirements file where needed.

## Running the work

For Assignment 1, the Python scripts can be run from a terminal or Anaconda Prompt:

```bash
python q1_gradient_descent.py
python q2_train_models.py
python q2_evaluation.py
```

For Assignment 2 and the final project, the notebook files can be opened and run in JupyterLab:

```bash
jupyter lab
```

When using a CUDA-enabled GPU, the notebooks/scripts will use GPU acceleration when available.

## Notes

Large generated files, model checkpoints, local cache files, and temporary notebook folders are not intended to be included in the repository. The saved reports, plots, and metric files are included to make the results easy to review without rerunning all experiments.

## Authors

- Mohamad Rami Karkour (Assignments and Final Project)
- Sahel Nasrullah (Final Project)
- Muhammad Haseeb Muslim (Final Project)
