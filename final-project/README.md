# Email Spam Detection System

A robust email spam detection system that combines three supervised machine learning algorithms: Hidden Markov Model (HMM), Support Vector Machine (SVM), and Naive Bayes (NB). The system provides detailed visualizations of model performance and training progress.

## Features

- **Multiple ML Models**: Utilizes three supervised learning algorithms:
  - Hidden Markov Model (HMM) for sequence-based analysis
  - Support Vector Machine (SVM) for high-dimensional classification
  - Naive Bayes (NB) for probabilistic classification

- **Comprehensive Training Visualizations**:
  - Individual learning curves for each model
  - ROC curves comparing model performance
  - Confusion matrices for detailed error analysis
  - Model accuracy comparison charts
  - Probability distribution visualization

- **Advanced Prediction System**:
  - Weighted probability combination from all three models
  - HMM-based sequence analysis
  - NB probability scoring
  - SVM classification with probability estimates
  - Interactive probability distribution graphs

## Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- pandas
- scikit-learn
- hmmlearn
- beautifulsoup4
- matplotlib
- seaborn
- tqdm
- joblib

## Project Structure

```
final-project/
├── dataset/
│   ├── ham/         # Non-spam email samples
│   └── spam/        # Spam email samples
├── models/          # Saved model files
├── visualizations/  # Generated graphs and charts
├── run_spam_detector.py    # Main training script
└── local_email_tester.py   # Testing script
```

## Usage

### Training the Models

To train the models with your dataset:

```bash
python run_spam_detector.py --data_dir ./dataset --train
```

This will:
1. Train all three models (HMM, SVM, NB)
2. Generate training visualizations
3. Save the trained models to the `models/` directory

### Testing with Local Emails

To test the system with local email files:

```bash
python local_email_tester.py
```

## Model Details

### Hidden Markov Model (HMM)
- Analyzes email content as a sequence of characters
- Captures temporal patterns in spam vs. non-spam emails
- Provides probability scores based on sequence likelihood

### Support Vector Machine (SVM)
- Uses TF-IDF features for classification
- Linear kernel for efficient processing
- Provides probability estimates for classification confidence

### Naive Bayes (NB)
- Probabilistic classifier based on word frequencies
- Fast and efficient for text classification
- Provides probability scores for spam likelihood

## Visualizations

The system generates several visualizations during training:

1. **Learning Curves**:
   - Individual curves for HMM, SVM, and NB
   - Shows training progress and model convergence

2. **ROC Curves**:
   - Compares performance of all three models
   - Shows trade-off between true positive and false positive rates

3. **Confusion Matrices**:
   - Detailed error analysis for each model
   - Shows true positives, false positives, true negatives, and false negatives

4. **Probability Distribution**:
   - 3D visualization of model predictions
   - Shows relationship between HMM, NB, and SVM probabilities

## Prediction Process

The system combines predictions from all three models:

1. HMM generates sequence-based probability
2. NB calculates word-based probability
3. SVM provides classification probability
4. Final prediction uses weighted combination:
   - NB: 40-60% weight (higher when predicting ham)
   - SVM: 25-35% weight
   - HMM: 5-15% weight
   - Additional features: 10% weight

## Contributing

Feel free to submit issues and enhancement requests! 