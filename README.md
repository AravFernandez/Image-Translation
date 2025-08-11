# Image-Translation

**Image-Translation** is a PyTorch-based implementation of the Pix2Pix conditional GAN, designed for image-to-image translation between satellite imagery and map views. The project supports translation in both directions—satellite-to-map and map-to-satellite—and includes functionality for dataset preparation, model training, checkpoints, standalone image translation, and **explainable AI (XAI)** integration using **LIME**.

## Features

- **Bidirectional translation**: Convert satellite images to map-style visuals, or generate plausible satellite imagery from map images.  
- **Standard Pix2Pix architecture**: Follows the original Pix2Pix paper’s generator–discriminator design.  
- **Checkpointing & visual feedback**:
  - Saves model weights every 10 epochs.
  - Generates sample outputs every 10 epochs for visual comparison.  
- **Standalone image translation**: Easily translate single input images with preprocessing & postprocessing built in.  
- **Support for creative input**: Translate “unrealistic” or procedurally generated map layouts into plausible satellite imagery.  
- **Explainability with LIME**:
  - Analyze which regions of the input image most influence the model’s output.
  - Visualize interpretable explanations for model decisions.

---

## Table of Contents

1. [Installation](#installation)  
2. [Dataset Preparation](#dataset-preparation)  
3. [Model Training](#model-training)  
4. [Evaluating and Tracking Performance](#evaluating-and-tracking-performance)  
5. [Explainability (XAI)](#explainability-xai)  
6. [Standalone Image Translation](#standalone-image-translation)  
7. [Custom/External Inputs](#customexternal-inputs)  
8. [Results Gallery](#results-gallery)  
9. [References](#references)  
10. [License](#license)

---

## Installation

```bash
git clone https://github.com/AravFernandez/Image-Translation.git
cd Image-Translation
python3 -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
pip install -r requirements.txt
