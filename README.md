This repository contains the complete reproducibility code for the HoloTiny-AD 

FILE STRUCTURE:
---------------
- preprocessing.py: Complete data preprocessing pipeline with holographic feature 
  extraction (implements Equation 6 from paper)
- data_splitting.py: Reproducible data splitting and comprehensive model evaluation
- config.py: Centralized configuration management for all experiment parameters

QUICK START:
-----------
1. Install dependencies:
   pip install pandas numpy scikit-learn torch pyyaml

2. Run demonstration:
   python data_splitting.py
   (This executes the complete pipeline with sample data)

3. For custom datasets:
   from preprocessing import HolographicDataPreprocessor
   from data_splitting import DataSplitter, ModelEvaluator
   
   preprocessor = HolographicDataPreprocessor(random_state=42)
   processed_df, report = preprocessor.preprocess_data(your_df, target_column)

KEY FEATURES:
-------------
- Fixed random seeds (42) throughout for perfect reproducibility
- Stratified 70/30 train/test splits maintaining class distribution
- Holographic feature extraction (temporal, spatial, behavioral, correlation)
- All models from the paper: Decision Tree, Random Forest, CNN, TCN, Autoencoder, etc.
- Multi-device edge computing simulation support
- Automated metadata removal and categorical encoding

MODELS IMPLEMENTED:
-------------------
Traditional ML:
- Logistic Regression, Decision Tree, Random Forest
- K-Nearest Neighbors, Gradient Boosting, SVM

Deep Learning:
- 1D CNN (2 convolutional layers + classifier)
- Temporal Convolutional Network (TCN)
- Autoencoder for anomaly detection

TinyML:
- Lightweight Decision Tree (primary model for edge deployment)

REPRODUCIBILITY GUARANTEES:
--------------------------
- Consistent random seeds across Python, NumPy, and PyTorch
- Saved split indices for exact dataset replication
- YAML configuration files capture all experimental settings
- Detailed logging of all preprocessing steps
- Stratified sampling for maintained class distributions

EXPERIMENTAL SETUP:
------------------
- Tested with device counts: 3, 5, 7, 9, 12, 15
- Uses CIC-BCCC-NRC-ACI-IOT-2023 and CIC-IDS2017 datasets
- Implements exact model architectures and hyperparameters

