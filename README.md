# SciLens
### Multimodal Scientific Question Answering Through Visual and Textual Reasoning

**SciLens** is a multimodal AI research project investigating how textual and visual information can be combined to answer scientific questions. Using the ScienceQA-IMG dataset, the project compares text-only and multimodal deep learning architectures to understand how different fusion strategies influence scientific question-answering performance.

## Project Overview

Scientific questions can require both language understanding and visual interpretation of diagrams, illustrations, and other images.

SciLens investigates this challenge through four model configurations:

1. **Text-only BERT:** Uses question and answer-choice text without image information.
2. **Early Fusion V2:** Combines BERT and ViT representations through feature concatenation, with a frozen ViT encoder.
3. **Early Fusion V3:** Uses trainable upper layers of both encoders and combines textual and visual representations.
4. **Intermediate Fusion V1:** Uses text-to-image cross-attention to enable interaction between textual tokens and image patch representations.

The primary objective is to evaluate whether visual information improves answer prediction and compare the effectiveness of different multimodal fusion approaches.

## Research Objectives

- Establish a text-only scientific question-answering baseline.
- Develop early and intermediate multimodal fusion architectures.
- Evaluate the contribution of visual information to answer prediction.
- Compare model performance using consistent evaluation metrics.
- Examine classification errors and limitations across the evaluated models.

## Dataset

This project uses the **ScienceQA-IMG** dataset.

- **Source:** [Hugging Face – ScienceQA-IMG](https://huggingface.co/datasets/lmms-lab/ScienceQA-IMG)
- **Original training samples:** 6,218
- **Original validation samples:** 2,097
- **Original test samples:** 2,017
- **Original total:** 10,332

After data cleaning and cross-split duplicate removal, the working dataset contains:

| Split | Cleaned samples |
|---|---:|
| Training | 3,530 |
| Validation | 1,343 |
| Testing | 2,017 |
| **Total** | **6,890** |

The original test split was retained unchanged. Preprocessing includes text tokenisation, image transformation, answer-choice formatting, and the preparation of attention and option masks.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Text Encoder | BERT (`bert-base-uncased`) |
| Image Encoder | Vision Transformer (`google/vit-base-patch16-224`) |
| Dataset | Hugging Face Datasets |
| Model Libraries | Hugging Face Transformers |
| Experimentation | Jupyter Notebook |
| GPU Acceleration | Apple MPS / CUDA |

## Project Structure

```text
SciLens/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_text_only_baseline.ipynb
│   ├── 04_early_fusion.ipynb
│   ├── 05_intermediate_fusion.ipynb
│   └── 06_model_evaluation.ipynb
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