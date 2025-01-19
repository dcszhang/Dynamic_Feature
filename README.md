# Dynamic Feature

Dynamic Feature is a project based on the following paper:

**Dynamic Feature Fusion: Combining Global Graph Structures and Local Semantics for Blockchain Fraud Detection**

- Authors: Zhang Sheng, Liangliang Song, Yanbin Wang
- Published in: arXiv preprint arXiv:2501.02032, 2025
- [Paper PDF](https://arxiv.org/abs/2501.02032)

## Project Background

With the rapid development of blockchain technology, the widespread adoption of smart contracts in the financial sector has led to the frequent occurrence of new types of fraud. However, existing fraud detection methods have limitations in capturing both global structural patterns in transaction networks and local semantic relationships in transaction data. To address these challenges, this project proposes a dynamic feature fusion model that combines graph-based representation learning with semantic feature extraction for blockchain fraud detection.

**Key Contributions:**
1. Built a global graph representation model to capture account relationships and extracted local contextual features from transaction data.
2. Proposed a dynamic multimodal fusion mechanism to adaptively integrate global structural and local semantic information.
3. Developed a comprehensive data processing pipeline, including graph construction, temporal feature enhancement, and text preprocessing.
4. Demonstrated the effectiveness of the method on large-scale real-world blockchain datasets, achieving superior performance in accuracy, F1 score, and recall compared to existing benchmarks.

## Features Overview

- **Data Processing:**
  - Provides tools for data cleaning and formatting, such as removing URLs and user tags.
  - Supports efficient sparse matrix operations, suitable for large-scale datasets.

- **Training:**
  - Implements training workflows in `train.py` with support for hyperparameter tuning and model evaluation.

## Project Structure

```
Dynamic_Feature/
├── Dataset/                 # Dataset folder
├── dev.tsv                  # Validation data
├── test.tsv                 # Test data
├── train.tsv                # Training data
├── env_config.py            # Environment variable configuration
├── ETH_GBert.py             # Model definition and extension
├── train.py                 # Training script
├── utils.py                 # Data processing utility functions
├── pytorch_pretrained_bert/ # Pre-trained BERT model resources
└── README.md                # Project description
```

## Installation and Setup

### Requirements

- Python 3.8 or higher
- PyTorch 1.9 or higher
- Additional dependencies listed in `requirements.txt`

### Installation Steps

1. Clone the project repository:
   ```bash
   git clone https://github.com/your-repo/Dynamic_Feature.git
   cd Dynamic_Feature
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   - Create a `.env` file in the root directory with the following content:
     ```env
     GLOBAL_SEED=44
     TRANSFORMERS_OFFLINE=0
     HUGGING_LOCAL_MODEL_FILES_PATH="/path/to/local/models"
     ```

## Usage Instructions

### Data Preprocessing

- Convert raw text data into `train.tsv`, `dev.tsv`, and `test.tsv` formats.
- Use utility functions in `utils.py` for data cleaning and formatting.

### Model Training

1. Run the training script:
   ```bash
   python train.py 
   ```

2. Key parameters supported:
   - `--epochs`: Total number of training epochs.
   - `--batch_size`: Size of each training batch.
   - `--learning_rate`: Learning rate.

3. After training, the model and logs will be saved in the specified directory.

### Model Evaluation

- Test the model using the `test.tsv` file and generate a classification report:
  ```bash
  python train.py --test --model_path ./saved_model
  ```

## Contributors

If you have any suggestions or improvements for the project, feel free to reach out via Pull Requests or Issues.

## License

This project is licensed under the MIT License. For details, please refer to the LICENSE file.

