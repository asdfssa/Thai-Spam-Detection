# Thai Spam Detection System 🛡️

A comprehensive Thai language spam detection system built with machine learning and natural language processing.

## Features

- **Thai Text Processing**: Advanced tokenization and preprocessing using `pythainlp`
- **Multiple ML Models**: Random Forest, Naive Bayes, Logistic Regression, and SVM
- **Web Interface**: Interactive Streamlit UI with real-time predictions
- **Batch Analysis**: Process multiple messages at once
- **Visualization**: Comprehensive charts and performance metrics
- **Modular Design**: Clean separation of concerns with separate modules

## Project Structure

```
Project/
├── Dataset/
│   └── thai_spam_production_v4.csv    # Training dataset
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py          # Text preprocessing module
│   ├── model_training.py              # Model training module
│   ├── model_evaluation.py            # Evaluation and visualization
│   └── web_ui.py                      # Streamlit web interface
├── models/                            # Trained models (created after training)
├── results/                           # Evaluation results (created after training)
├── train_model.py                     # Main training script
├── run_web_ui.py                      # Web UI launcher
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Installation

1. Clone or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

First, train the spam detection model:

```bash
python train_model.py
```

This will:
- Load and preprocess the Thai dataset
- Train multiple machine learning models
- Select the best performing model
- Perform hyperparameter tuning
- Generate evaluation reports and visualizations
- Save the trained model and preprocessing components

### 2. Launch Web Interface

After training, start the web UI:

```bash
python run_web_ui.py
```

Or directly with Streamlit:

```bash
streamlit run src/web_ui.py
```

The web interface provides:
- **Single Message Analysis**: Analyze individual messages in real-time
- **Batch Analysis**: Upload CSV files for bulk processing
- **Model Information**: View model details and performance metrics

## Web Interface Features

### 🔍 Single Message Analysis
- Input Thai text for spam detection
- Real-time prediction with confidence scores
- Detailed text preprocessing analysis
- Visual probability distribution

### 📁 Batch Analysis
- Upload CSV files with multiple messages
- Bulk processing with progress tracking
- Summary statistics and visualizations
- Download results as CSV

### 🤖 Model Information
- Model architecture and details
- Performance metrics and evaluation reports
- Confusion matrix and classification reports

## Technical Details

### Data Preprocessing
- **Text Cleaning**: URL removal, special character filtering
- **Tokenization**: Thai word segmentation using `pythainlp`
- **Vectorization**: TF-IDF feature extraction
- **Normalization**: Text normalization and standardization

### Machine Learning Models
- **Random Forest**: Ensemble decision trees
- **Naive Bayes**: Probabilistic classifier
- **Logistic Regression**: Linear classification
- **SVM**: Support Vector Machine with probability estimates

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC Curve and AUC
- Confusion Matrix
- Cross-validation scores

## Dependencies

- `streamlit` - Web interface framework
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning library
- `pythainlp` - Thai natural language processing
- `matplotlib` - Plotting library
- `seaborn` - Statistical visualization
- `plotly` - Interactive charts
- `pillow` - Image processing

## Model Performance

After training, the system generates comprehensive evaluation reports:
- Performance metrics per class
- Confusion matrices
- ROC and Precision-Recall curves
- Cross-validation results

## File Descriptions

### Core Modules
- `data_preprocessing.py`: Handles Thai text cleaning, tokenization, and feature extraction
- `model_training.py`: Implements model training, comparison, and hyperparameter tuning
- `model_evaluation.py`: Provides comprehensive evaluation and visualization tools
- `web_ui.py`: Streamlit-based web interface with interactive features

### Scripts
- `train_model.py`: Main training pipeline that orchestrates the entire process
- `run_web_ui.py`: Launcher script for the web interface

## Troubleshooting

### Common Issues

1. **Model not found error**
   - Run `python train_model.py` first to train and save the model

2. **pythainlp installation issues**
   - Try: `pip install pythainlp --upgrade`

3. **Streamlit not starting**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

4. **Memory issues with large datasets**
   - Reduce `max_features` in TF-IDF vectorizer
   - Use smaller dataset samples for testing

## Contributing

Feel free to contribute improvements:
- Add new machine learning models
- Enhance text preprocessing
- Improve web interface features
- Add more visualization options

## License

This project is for educational and research purposes.

---

**Built with ❤️ using Python, Streamlit, and pythainlp**
