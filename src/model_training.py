"""
Model training module for Thai spam detection
Handles model training, evaluation, and saving
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import ThaiTextPreprocessor, load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpamDetectionModel:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.preprocessor = ThaiTextPreprocessor()
        
    def train_models(self, X_train, y_train):
        """Train multiple models and compare performance"""
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'NaiveBayes': MultinomialNB(),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Train model
            model.fit(X_train, y_train)
            
            results[name] = {
                'model': model,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            logger.info(f"{name} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def select_best_model(self, results):
        """Select the best model based on cross-validation score"""
        best_model_name = max(results.keys(), key=lambda x: results[x]['cv_score'])
        self.model = results[best_model_name]['model']
        self.model_name = best_model_name
        
        logger.info(f"Best model selected: {best_model_name}")
        logger.info(f"Best CV Score: {results[best_model_name]['cv_score']:.4f}")
        
        return best_model_name, results
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='RandomForest'):
        """Perform hyperparameter tuning for the selected model"""
        logger.info(f"Starting hyperparameter tuning for {model_type}...")
        
        if model_type == 'RandomForest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
            model = RandomForestClassifier(random_state=42)
            
        elif model_type == 'NaiveBayes':
            param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0]}
            model = MultinomialNB()
            
        elif model_type == 'LogisticRegression':
            param_grid = {'C': [0.1, 1, 10], 'penalty': ['l2']}
            model = LogisticRegression(random_state=42, max_iter=1000)
            
        elif model_type == 'SVM':
            # เพิ่มให้รองรับ SVM ด้วย เผื่อรอบไหน SVM ชนะ
            param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
            model = SVC(random_state=42, probability=True)
            
        else:
            logger.warning("Hyperparameter tuning not implemented for this model type")
            return None, None
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.model_name = model_type
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_params_, grid_search.best_score_
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model on unseen TEST data only"""
        y_pred = self.model.predict(X_test)
        
        # ป้องกัน Error กรณีที่โมเดลไม่มี predict_proba (เช่น LinearSVC)
        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(X_test)[:, 1]
        else:
            y_prob = None
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"\n=== Final Model Evaluation (On Unseen Data) ===")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    def save_model(self, model_path):
        if self.model:
            # สร้างโฟลเดอร์อัตโนมัติเพื่อป้องกัน FileNotFoundError
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, text):
        if not self.model or not self.preprocessor.vectorizer:
            return None
        
        cleaned_text = self.preprocessor.clean_text(text)
        tokens = self.preprocessor.tokenize_thai(cleaned_text)
        processed_text = ' '.join(tokens)
        
        text_vector = self.preprocessor.vectorizer.transform([processed_text])
        prediction = self.model.predict(text_vector)[0]
        probability = self.model.predict_proba(text_vector)[0]
        
        label = self.preprocessor.label_encoder.inverse_transform([prediction])[0]
        
        # ค้นหา Index ที่แท้จริงของคำว่า 'spam' อย่างปลอดภัย
        classes = list(self.preprocessor.label_encoder.classes_)
        try:
            spam_idx = classes.index('spam')
            spam_prob = probability[spam_idx]
        except ValueError:
            # สำรองไว้เผื่อใช้คำอื่นแทน spam
            spam_prob = probability[1] if len(probability) > 1 else probability[0]
            
        return {
            'label': label,
            'confidence': max(probability),
            'spam_probability': spam_prob
        }

def plot_confusion_matrix(cm, labels, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix on Unseen Data')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        # สร้างโฟลเดอร์อัตโนมัติเพื่อป้องกัน FileNotFoundError
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # เปลี่ยนเป็น close เพื่อไม่ให้สคริปต์ค้าง

if __name__ == "__main__":
    trainer = SpamDetectionModel()
    
    # โหลดไฟล์ dataset
    df = load_dataset("Dataset/thai_spam_production_v4.csv") 
    
    if df is not None:
        processed_df = trainer.preprocessor.preprocess_dataset(df)
        X_all = trainer.preprocessor.create_tfidf_vectorizer(processed_df['processed_text'])
        y_all = processed_df['encoded_label']
        
        # 1. แบ่งข้อมูลก่อนทำอย่างอื่นทั้งหมด!
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
        
        # 2. Train และ Tune โดยใช้ "เฉพาะ X_train"
        results = trainer.train_models(X_train, y_train)
        best_model_name, results = trainer.select_best_model(results)
        trainer.hyperparameter_tuning(X_train, y_train, best_model_name)
        
        # 3. Evaluate โดยใช้ "เฉพาะ X_test" ที่โมเดลไม่เคยเห็นมาก่อน
        evaluation = trainer.evaluate_model(X_test, y_test)
        
        # 4. บันทึกผลและโมเดล (ปรับ Path ให้เซฟลงโฟลเดอร์ปัจจุบัน)
        labels = trainer.preprocessor.label_encoder.classes_
        plot_confusion_matrix(evaluation['confusion_matrix'], labels, "results/confusion_matrix.png")
        
        trainer.save_model("models/spam_detection_model.pkl")
        
        # เผื่อไว้ว่าฟังก์ชันใน preprocessor ไม่ได้สร้างโฟลเดอร์ให้
        os.makedirs("models", exist_ok=True)
        trainer.preprocessor.save_preprocessor(
            "models/vectorizer.pkl",
            "models/label_encoder.pkl"
        )
        logger.info("Training pipeline completed successfully!")