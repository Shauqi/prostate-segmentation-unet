# Prostate Segmentation using U-Net

This repository contains the implementation of a U-Net model for prostate segmentation. The project includes data preprocessing, model training, evaluation, and inference scripts.

## Dataset

The dataset used in this project is sourced from the [SAML Dataset](https://liuquande.github.io/SAML/), which contains a large collection of annotated medical images for semantic segmentation. These images are specifically tailored for machine learning models to learn the segmentation of anatomical structures in medical scans.

## U-Net Model

The U-Net model is a convolutional neural network originally designed for biomedical image segmentation. The architecture is structured as a U-shaped network to efficiently learn from a small number of images while achieving precise localizations. Its design includes a contracting path to capture context and an expansive path that enables precise localization, making it highly effective for tasks like medical image segmentation.

## Repository Structure

- `dataloader.py`: Script for loading and preprocessing the data.
- `train.ipynb`: Jupyter notebook for training the U-Net model.
- `evaluation.ipynb`: Jupyter notebook for evaluating the trained model.
- `inference.ipynb`: Jupyter notebook for running inference using the trained model.
- `main.py`: Main script to run the training and evaluation pipeline.
- `model.py`: Definition of the U-Net model architecture.
- `pre_process.ipynb`: Jupyter notebook for data preprocessing.
- `pre_process.py`: Script for data preprocessing.
- `README.md`: Documentation for the project.
- `test.py`: Script for testing the model.
- `utils.py`: Utility functions used across the project.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch
- NumPy
- Matplotlib
- Other dependencies listed in `requirements.txt` (if available)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd prostate-segmentation-unet
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the U-Net model, use the `train.ipynb` notebook or run the `main.py` script.

### Evaluation

Evaluate the trained model using the `evaluation.ipynb` notebook.

### Inference

Run inference on new data using the `inference.ipynb` notebook.

## Results

The U-Net model achieves high accuracy in segmenting prostate regions from medical images. Refer to the `evaluation.ipynb` notebook for detailed metrics and visualizations.

Here is a visualization of the model's segmentation output compared to the ground truth:

![Segmentation Results](result/Figure_2.png)

## Dataset Links

- **Main Dataset:** [SAML Dataset Link](https://drive.google.com/file/d/1TtrjnlnJ1yqr5m4LUGMelKTQXtvZaru-/view?usp=sharing)
- **Processed Dataset:** [Processed Dataset](https://drive.google.com/file/d/16Xrat8Sop6E0B6eK4TMTAz4Wu6VaAPPD/view?usp=sharing)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- The U-Net architecture is based on the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger et al.
- Special thanks to the contributors and open-source libraries used in this project.