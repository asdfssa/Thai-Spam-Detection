"""
Model evaluation module for Thai spam detection
Provides comprehensive evaluation metrics and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc,
    precision_recall_curve
)
import logging
from model_training import SpamDetectionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.model = SpamDetectionModel()
        
    def load_model_and_components(self, model_path, vectorizer_path, encoder_path):
        """Load trained model and preprocessing components"""
        model_loaded = self.model.load_model(model_path)
        preprocessor_loaded = self.model.preprocessor.load_preprocessor(vectorizer_path, encoder_path)
        
        if model_loaded and preprocessor_loaded:
            logger.info("Model and components loaded successfully")
            return True
        else:
            logger.error("Failed to load model or components")
            return False
    
    def comprehensive_evaluation(self, X_test, y_test):
        """Perform comprehensive model evaluation"""
        logger.info("Starting comprehensive evaluation...")
        
        # Predictions
        y_pred = self.model.model.predict(X_test)
        y_prob = self.model.model.predict_proba(X_test)[:, 1]
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None)
        recall_per_class = recall_score(y_test, y_pred, average=None)
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC curve (for binary classification)
        if len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall_curve, precision_curve)
        else:
            fpr, tpr, roc_auc = None, None, None
            precision_curve, recall_curve, pr_auc = None, None, None
        
        results = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'pr_auc': pr_auc
        }
        
        # Log results
        logger.info(f"Evaluation Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision (Weighted): {precision:.4f}")
        logger.info(f"Recall (Weighted): {recall:.4f}")
        logger.info(f"F1-Score (Weighted): {f1:.4f}")
        
        if roc_auc is not None:
            logger.info(f"ROC AUC: {roc_auc:.4f}")
            logger.info(f"PR AUC: {pr_auc:.4f}")
        
        return results
    
    def plot_evaluation_metrics(self, results, save_dir="../results"):
        """Create comprehensive evaluation plots"""
        # Create directory if it doesn't exist
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix(results, f"{save_dir}/confusion_matrix.png")
        
        # 2. ROC Curve (if binary classification)
        if results['roc_auc'] is not None:
            self._plot_roc_curve(results, f"{save_dir}/roc_curve.png")
            self._plot_precision_recall_curve(results, f"{save_dir}/pr_curve.png")
        
        # 3. Metrics Comparison
        self._plot_metrics_comparison(results, f"{save_dir}/metrics_comparison.png")
        
        # 4. Prediction Confidence Distribution
        self._plot_confidence_distribution(results, f"{save_dir}/confidence_distribution.png")
        
        logger.info(f"Evaluation plots saved to {save_dir}")
    
    def _plot_confusion_matrix(self, results, save_path):
        """Plot confusion matrix with percentages"""
        cm = results['confusion_matrix']
        labels = self.model.preprocessor.label_encoder.classes_
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax2)
        ax2.set_title('Confusion Matrix (Percentages)')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_roc_curve(self, results, save_path):
        """Plot ROC curve"""
        fpr = results['fpr']
        tpr = results['tpr']
        roc_auc = results['roc_auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_precision_recall_curve(self, results, save_path):
        """Plot Precision-Recall curve"""
        precision_curve = results['precision_curve']
        recall_curve = results['recall_curve']
        pr_auc = results['pr_auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_metrics_comparison(self, results, save_path):
        """Plot per-class metrics comparison"""
        labels = self.model.preprocessor.label_encoder.classes_
        
        precision = results['precision_per_class']
        recall = results['recall_per_class']
        f1 = results['f1_per_class']
        
        x = np.arange(len(labels))
        width = 0.25
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
        plt.bar(x, recall, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Per-Class Performance Metrics')
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            plt.text(i - width, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=9)
            plt.text(i, r + 0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=9)
            plt.text(i + width, f + 0.01, f'{f:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_confidence_distribution(self, results, save_path):
        """Plot prediction confidence distribution"""
        y_test = results['y_test']
        y_prob = results['y_prob']
        
        plt.figure(figsize=(12, 5))
        
        # Separate probabilities for correct and incorrect predictions
        y_pred = results['y_pred']
        correct_mask = (y_test == y_pred)
        
        plt.subplot(1, 2, 1)
        plt.hist(y_prob[correct_mask], bins=50, alpha=0.7, label='Correct Predictions', color='green')
        plt.hist(y_prob[~correct_mask], bins=50, alpha=0.7, label='Incorrect Predictions', color='red')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution by Prediction Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confidence threshold analysis
        plt.subplot(1, 2, 2)
        thresholds = np.arange(0.1, 1.0, 0.1)
        accuracies = []
        
        for threshold in thresholds:
            pred_at_threshold = (y_prob >= 0.5).astype(int)  # Assuming binary classification
            acc = accuracy_score(y_test, pred_at_threshold)
            accuracies.append(acc)
        
        plt.plot(thresholds, accuracies, marker='o')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Confidence Threshold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self, results, save_path="../results/evaluation_report.txt"):
        """Generate detailed evaluation report"""
        report = []
        report.append("=" * 60)
        report.append("THAI SPAM DETECTION MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall metrics
        report.append("OVERALL PERFORMANCE METRICS:")
        report.append(f"Accuracy: {results['accuracy']:.4f}")
        report.append(f"Precision (Weighted): {results['precision_weighted']:.4f}")
        report.append(f"Recall (Weighted): {results['recall_weighted']:.4f}")
        report.append(f"F1-Score (Weighted): {results['f1_weighted']:.4f}")
        report.append("")
        
        # Per-class metrics
        labels = self.model.preprocessor.label_encoder.classes_
        report.append("PER-CLASS PERFORMANCE METRICS:")
        for i, label in enumerate(labels):
            report.append(f"\n{label.upper()}:")
            report.append(f"  Precision: {results['precision_per_class'][i]:.4f}")
            report.append(f"  Recall: {results['recall_per_class'][i]:.4f}")
            report.append(f"  F1-Score: {results['f1_per_class'][i]:.4f}")
        
        # ROC AUC (if applicable)
        if results['roc_auc'] is not None:
            report.append("")
            report.append("BINARY CLASSIFICATION METRICS:")
            report.append(f"ROC AUC: {results['roc_auc']:.4f}")
            report.append(f"PR AUC: {results['pr_auc']:.4f}")
        
        # Confusion matrix
        report.append("")
        report.append("CONFUSION MATRIX:")
        cm = results['confusion_matrix']
        report.append("Predicted →")
        header = "True ↓\t" + "\t".join(labels)
        report.append(header)
        for i, label in enumerate(labels):
            row = f"{label}\t" + "\t".join(map(str, cm[i]))
            report.append(row)
        
        # Detailed classification report
        report.append("")
        report.append("DETAILED CLASSIFICATION REPORT:")
        report.append(str(classification_report(results['y_test'], results['y_pred'])))
        
        # Save report
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Evaluation report saved to {save_path}")
        return '\n'.join(report)

if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    
    # Load model and components
    if evaluator.load_model_and_components(
        "../models/spam_detection_model.pkl",
        "../models/vectorizer.pkl", 
        "../models/label_encoder.pkl"
    ):
        # Load test data
        from data_preprocessing import load_dataset
        df = load_dataset("../Dataset/thai_spam_production_v4.csv")
        if df is not None:
            processed_df = evaluator.model.preprocessor.preprocess_dataset(df)
            X = evaluator.model.preprocessor.create_tfidf_vectorizer(processed_df['processed_text'])
            y = processed_df['encoded_label']
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Evaluate
            results = evaluator.comprehensive_evaluation(X_test, y_test)
            
            # Generate plots and report
            evaluator.plot_evaluation_metrics(results)
            report = evaluator.generate_evaluation_report(results)
            print(report)
