"""
Main training script for Thai Spam Detection Model
Run this script to train and evaluate the model
"""

import sys
import os
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import load_dataset
from src.model_training import SpamDetectionModel
from src.model_evaluation import ModelEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main training pipeline"""
    logger.info("=" * 60)
    logger.info("STARTING THAI SPAM DETECTION MODEL TRAINING")
    logger.info("=" * 60)
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    try:
        # Initialize trainer
        trainer = SpamDetectionModel()
        
        # Load dataset
        logger.info("# Load dataset")
        df = load_dataset("Dataset/thai_spam_production_v4.csv")
        
        if df is None:
            logger.error("Failed to load dataset. Exiting...")
            return
        
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")
        
        # Preprocess data
        logger.info("Preprocessing data...")
        processed_df = trainer.preprocessor.preprocess_dataset(df)
        
        # Create TF-IDF features
        logger.info("Creating TF-IDF features...")
        X = trainer.preprocessor.create_tfidf_vectorizer(processed_df['processed_text'])
        y = processed_df['encoded_label']
        
        # Train multiple models
        logger.info("Training multiple models...")
        results = trainer.train_models(X, y)
        
        # Select best model
        logger.info("Selecting best model...")
        best_model_name, results = trainer.select_best_model(results)
        
        # Hyperparameter tuning
        logger.info("Performing hyperparameter tuning...")
        best_params, best_score = trainer.hyperparameter_tuning(X, y, best_model_name)
        
        # Final evaluation
        logger.info("Performing final evaluation...")
        evaluation = trainer.evaluate_model(X, y)
        
        # Save model and components
        logger.info("Saving model and components...")
        trainer.save_model("models/spam_detection_model.pkl")
        trainer.preprocessor.save_preprocessor(
            "models/vectorizer.pkl",
            "models/label_encoder.pkl"
        )
        
        # Generate evaluation report
        logger.info("Generating evaluation report...")
        evaluator = ModelEvaluator()
        evaluator.model = trainer
        evaluator.load_model_and_components(
            "models/spam_detection_model.pkl",
            "models/vectorizer.pkl",
            "models/label_encoder.pkl"
        )
        
        # Split data for final evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Comprehensive evaluation
        eval_results = evaluator.comprehensive_evaluation(X_test, y_test)
        evaluator.plot_evaluation_metrics(eval_results)
        report = evaluator.generate_evaluation_report(eval_results)
        
        # Save processed data
        processed_df.to_csv("Dataset/processed_data.csv", index=False)
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"Best CV Score: {best_score:.4f}")
        logger.info(f"Final Test Accuracy: {evaluation['accuracy']:.4f}")
        logger.info("=" * 60)
        
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"✅ Dataset: {df.shape[0]} messages")
        print(f"✅ Best Model: {best_model_name}")
        print(f"✅ Cross-Validation Score: {best_score:.4f}")
        print(f"✅ Test Accuracy: {evaluation['accuracy']:.4f}")
        print(f"✅ Model saved to: models/spam_detection_model.pkl")
        print(f"✅ Evaluation report: results/evaluation_report.txt")
        print(f"✅ Visualization plots: results/")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
