# SciLens
### Multimodal Scientific Question Answering Through Visual and Textual Reasoning

**SciLens** is a multimodal AI research project that explores how combining textual and visual information can improve scientific question answering. Using the ScienceQA-IMG dataset, the project investigates how different multimodal fusion strategies influence a model's ability to understand questions, interpret images, and select correct answers.

## Project Overview

Scientific questions often require more than textual understanding. Some questions rely on diagrams, illustrations, charts, or other visual information to reach the correct answer.

SciLens explores this challenge by developing and comparing three deep learning approaches:

1. **Text-only Baseline (BERT):** Uses textual information from questions and answer choices without considering images.
2. **Early Fusion:** Combines text representations from BERT with visual representations from Vision Transformer (ViT) before answer prediction.
3. **Intermediate Fusion:** Explores interaction between textual and visual representations through attention-based fusion.

The objective is to evaluate whether multimodal learning provides an advantage over text-only approaches and understand how different fusion strategies affect performance.

## Research Objectives

- Develop a reliable text-only baseline for scientific question answering.
- Investigate early and intermediate multimodal fusion strategies.
- Evaluate the contribution of visual information to answer prediction.
- Compare model performance using consistent experimental settings.
- Analyse model behaviour across different scientific subjects and grade levels.

## Dataset

This project uses the **ScienceQA-IMG** dataset from Hugging Face.

- **Source:** [lmms-lab/ScienceQA-IMG](https://huggingface.co/datasets/lmms-lab/ScienceQA-IMG)
- **Original training samples:** 6,218
- **Original validation samples:** 2,097
- **Original test samples:** 2,017
- **Total:** 10,332 samples

The dataset contains scientific questions, answer choices, images, labels, and additional contextual information.

Data preprocessing includes text tokenisation, image transformation, answer formatting, and cross-split duplicate checking.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Text Encoder | BERT |
| Image Encoder | Vision Transformer (ViT) |
| Dataset Management | Hugging Face Datasets |
| Model Libraries | Hugging Face Transformers |
| Experimentation | Jupyter Notebook |
| GPU Acceleration | Apple MPS / Google Colab CUDA |

## Project Structure

```text
SciLens/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_text_only_baseline.ipynb
│   └── 04_early_fusion.ipynb
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│
├── 01_load_dataset.py
├── 02_dataset_analysis.py
├── test_gpu.py
│
├── .gitignore
└── README.md
```

Large datasets, processed tensors, model checkpoints, and generated results are excluded from the Git repository.

## Experimental Methodology

The experiments follow a comparative approach:

**1. Data Exploration and Preparation**

Explore the dataset, inspect image and text distributions, identify missing or invalid records, and remove cross-split duplicates.

**2. Text-only Baseline**

Develop a BERT-based model using question and answer-choice representations to establish a baseline.

**3. Early Fusion**

Extract textual and visual representations independently, concatenate them, and use a trainable classifier to predict the correct answer.

**4. Intermediate Fusion**

Investigate attention-based interaction between question representations and image patch embeddings.

**5. Model Evaluation**

Compare models using validation and test accuracy, cross-entropy loss, and additional evaluation measures.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/SciLens.git
cd SciLens
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision transformers datasets accelerate scikit-learn tqdm jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Load the Dataset

The dataset can be loaded directly from Hugging Face:

```python
from datasets import load_dataset

dataset = load_dataset("lmms-lab/ScienceQA-IMG")
```

### 6. Run the Experiments

Execute the notebooks in numerical order, beginning with dataset exploration and preprocessing.

For GPU training, the notebooks can also be executed using Google Colab with a compatible GPU runtime.

## Evaluation Metrics

The project investigates model performance through:

- Accuracy
- Cross-entropy loss
- Calibration
- Subject-wise performance
- Grade-wise performance
- Image-shuffling diagnostics

Validation data is used for model selection, while the test set is reserved for final evaluation.

## Research Scope

SciLens focuses on understanding the effectiveness of visual and textual information in scientific question answering. Rather than assuming that multimodal models will always outperform text-only models, the project evaluates the actual contribution of visual features under different fusion architectures.

## Project Status

**Status:** Research and development in progress.

The text-only baseline and initial early-fusion experiments have been developed. Further experiments focus on improving multimodal fusion and conducting comparative evaluation.

## Acknowledgements

- [ScienceQA-IMG Dataset](https://huggingface.co/datasets/lmms-lab/ScienceQA-IMG)
- [Hugging Face](https://huggingface.co/)
- [PyTorch](https://pytorch.org/)
- [Google Colab](https://colab.research.google.com/)

---

**SciLens — Seeing science. Understanding questions. Reasoning with AI.**
