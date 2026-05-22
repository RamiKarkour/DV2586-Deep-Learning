# AI-Based Resume Screening and Job Matching

This repository contains a deep learning project for matching candidate resumes with relevant job postings using natural language processing (NLP), information retrieval methods, and transformer-based text embeddings.

The project compares classical keyword-based ranking methods with pretrained and fine-tuned Sentence-BERT models. The final proposed method combines fine-tuned Sentence-BERT similarity with a lightweight skill-overlap reranking signal.

---

## Project Overview

The system takes resume text and job posting text as input, converts them into numerical representations, and ranks job postings according to their relevance to each resume.

The general pipeline is:

```text
Resume text → text cleaning → embedding/vector representation
Job text → text cleaning → embedding/vector representation
Resume/job comparison → ranking scores → top job recommendations
```

The project evaluates five methods:

1. **TF-IDF + cosine similarity**
2. **BM25 ranking**
3. **Pretrained Sentence-BERT**
4. **Fine-tuned Sentence-BERT**
5. **Hybrid Fine-tuned Sentence-BERT + Skill Overlap**

The final proposed hybrid score is:

```text
Hybrid score = 0.85 × Fine-tuned SBERT similarity + 0.15 × Skill overlap
```

---

## Datasets

This project uses two Kaggle datasets:

1. **Resume dataset**  
   `snehaanbhawal/resume-dataset`

2. **LinkedIn job postings dataset**  
   `arshkon/linkedin-job-postings`

The raw datasets are not included in this repository because they are large. They should be downloaded manually from Kaggle and placed in the folder structure shown below.

Expected raw data structure:

```text
data/
└── raw/
    ├── resume-dataset/
    │   └── resume/
    │       └── Resume.csv
    │
    └── linkedin-job-postings/
        ├── postings.csv
        ├── jobs/
        │   └── job_skills.csv
        └── mappings/
            └── skills.csv
```

---

## Repository Structure

```text
DL_Final_Project/
│
├── data/
│   ├── raw/                  # Original Kaggle datasets, not included in GitHub
│   └── processed/            # Selected processed files and category mappings
│
├── notebooks/
│   └── final_project_resume_job_matching.ipynb
│
├── outputs/
│   ├── metrics/              # Evaluation tables
│   ├── plots/                # Figures used for analysis and presentation
│   └── results/              # Sample rankings and qualitative analysis
│
├── report/                   # Report files
│
├── requirements.txt
├── .gitignore
└── README.md
```

Large generated files such as saved models, embeddings, raw datasets, and ZIP files are intentionally excluded from GitHub.

---

## Methods

### 1. TF-IDF + Cosine Similarity

A classical NLP baseline that represents resumes and job postings using TF-IDF features and compares them with cosine similarity.

### 2. BM25

A ranking baseline commonly used in information retrieval. It ranks job postings based on query-term relevance using resume text as the query.

### 3. Pretrained Sentence-BERT

A transformer-based semantic similarity baseline. The model converts resumes and job postings into dense sentence embeddings and compares them using cosine similarity.

### 4. Fine-tuned Sentence-BERT

The main deep learning method. Sentence-BERT is fine-tuned using positive resume-job pairs constructed from category-aligned matches. The training uses contrastive learning through `MultipleNegativesRankingLoss`.

### 5. Hybrid Reranking

The final proposed method combines fine-tuned SBERT similarity with a skill-overlap score. This adds an explicit skill/category matching signal on top of semantic similarity.

---

## Evaluation Setup

The datasets do not include manually labeled resume-job relevance pairs. Therefore, relevance labels were constructed using resume categories and job title/skill/category keyword mapping.

A job is treated as relevant to a resume if the resume category appears in the job's matched categories.

Example:

```text
Resume category: HR
Job title: Human Resources Generalist
Matched category: HR
Result: Relevant
```

The final evaluation used:

```text
1,935 evaluation resumes
6,879 evaluation jobs
17 valid resume/job categories
```

A stratified split was created:

```text
Train resumes: 1,354
Validation resumes: 194
Test resumes: 387
```

The final comparison was performed on the same test set for all methods.

---

## Metrics

The following ranking metrics were used:

- **Top-1 Accuracy**
- **Top-3 Accuracy**
- **Top-5 Accuracy**
- **Top-10 Accuracy**
- **Precision@K**
- **Mean Reciprocal Rank (MRR)**

---

## Results

Final test-set results:

| Method | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy | Top-10 Accuracy | MRR |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Cosine Similarity | 0.576 | 0.682 | 0.713 | 0.780 | 0.646 |
| BM25 | 0.499 | 0.636 | 0.698 | 0.762 | 0.593 |
| Pretrained Sentence-BERT | 0.615 | 0.757 | 0.791 | 0.819 | 0.693 |
| Fine-tuned Sentence-BERT | 0.788 | 0.871 | 0.899 | 0.904 | 0.831 |
| Hybrid Fine-tuned SBERT + Skill Overlap | 0.811 | 0.876 | 0.894 | 0.907 | 0.849 |

The hybrid model achieved the best overall performance for Top-1 Accuracy, Top-3 Accuracy, Top-10 Accuracy, and MRR. Fine-tuned Sentence-BERT achieved the best Top-5 Accuracy.

---

## Key Output Files

Useful output files include:

```text
outputs/metrics/final_model_comparison_with_hybrid.csv
outputs/metrics/final_model_comparison_rounded.csv
outputs/metrics/hybrid_weight_ablation.csv
outputs/metrics/hybrid_per_category_top1_accuracy.csv

outputs/plots/model_comparison_test_set.png
outputs/plots/top1_mrr_comparison.png
outputs/plots/hybrid_weight_ablation.png
outputs/plots/finetuned_sbert_training_loss_smoothed.png
outputs/plots/hybrid_per_category_top1_accuracy.png

outputs/results/hybrid_correct_top1_examples.csv
outputs/results/hybrid_failed_top1_examples.csv
outputs/results/hybrid_improved_over_finetuned_cases.csv
outputs/results/hybrid_worse_than_finetuned_cases.csv
```

---

## Installation

Create a Python environment:

```bash
conda create -n dl-final python=3.10 -y
conda activate dl-final
```

Install required packages:

```bash
pip install -r requirements.txt
```

For GPU training, install PyTorch with CUDA according to the official PyTorch command for your system.

Example for a compatible Windows/NVIDIA CUDA setup:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

Check GPU availability:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## How to Run

1. Download the two Kaggle datasets.
2. Place the files inside the expected `data/raw/` structure.
3. Install the dependencies.
4. Open the notebook:

```text
notebooks/final_project_resume_job_matching.ipynb
```

5. Run the notebook from top to bottom.

The notebook performs:

- raw data loading
- text cleaning
- job skill merging
- category mapping
- train/validation/test split
- baseline ranking methods
- pretrained SBERT evaluation
- fine-tuned SBERT training
- hybrid reranking
- ablation study
- plots and qualitative analysis

---

## Limitations

The project has several limitations:

- Relevance labels were generated using category and keyword mapping, not human expert annotation.
- Some categories are broader or more ambiguous than others, which affects evaluation quality.
- LinkedIn skill labels are limited and sometimes broad.
- The hybrid skill-overlap component uses simple overlap rather than a full skill extraction model.
- The evaluation focuses on ranking quality, not fairness, bias, or real hiring decisions.


---

## Authors

- Mohamad Rami Karkour
- Sahel Nasrullah
- Muhammad Haseeb Muslim