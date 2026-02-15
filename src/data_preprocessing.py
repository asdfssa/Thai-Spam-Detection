"""
Data preprocessing module for Thai spam detection
Handles text cleaning, tokenization, and feature extraction
"""

import pandas as pd
import re
import pythainlp
from pythainlp import word_tokenize
from pythainlp.corpus.common import thai_words
from pythainlp.util import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThaiTextPreprocessor:
    def __init__(self):
        self.custom_words = set(thai_words())
        self.vectorizer = None
        self.label_encoder = None
        
    def clean_text(self, text):
        """Clean and normalize Thai text"""
        if not isinstance(text, str):
            text = str(text)
        
        # 1. เปลี่ยน URLs ต่างๆ ให้กลายเป็นแท็ก __URL__ แทนที่จะลบทิ้ง
        text = re.sub(r'http[s]?://\S+', '__URL__', text)
        text = re.sub(r'www\.\S+', '__URL__', text)
        text = re.sub(r'bit\.ly\S+', '__URL__', text)
        
        # 2. ลบอักขระพิเศษ แต่ **อนุญาตให้เก็บภาษาอังกฤษ (A-Za-z)** ตัวเลข ภาษาไทย และ _ ไว้
        # เพราะคำอย่าง OTP, Line, Facebook เป็นคีย์เวิร์ดสำคัญของสแปม
        text = re.sub(r'[^\u0E00-\u0E7FA-Za-z\s0-9.,!?_]', '', text)
        
        # Normalize text (จัดการสระซ้อน ทันฑฆาต ฯลฯ)
        text = normalize(text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def tokenize_thai(self, text):
        """Tokenize Thai text using pythainlp"""
        try:
            # ใช้ engine 'newmm' ซึ่งตัดคำไทยได้ดีที่สุด
            tokens = word_tokenize(text, engine='newmm')
            
            # เก็บ token ที่มีความยาว > 1 หรือเป็นตัวเลขเดี่ยวๆ ก็เก็บไว้
            tokens = [token for token in tokens if len(token) > 1 or token.isnumeric()]
            return tokens
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            return []
    
    def preprocess_dataset(self, df, text_column='message', label_column='label'):
        """Preprocess the entire dataset"""
        logger.info("Starting data preprocessing...")
        
        # Clean text
        logger.info("Cleaning text data...")
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Tokenize
        logger.info("Tokenizing Thai text...")
        df['tokens'] = df['cleaned_text'].apply(self.tokenize_thai)
        
        # Join tokens back to text for vectorization
        df['processed_text'] = df['tokens'].apply(lambda tokens: ' '.join(tokens))
        
        # Remove empty rows
        df = df[df['processed_text'].str.len() > 0].copy()
        
        # Encode labels
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            df['encoded_label'] = self.label_encoder.fit_transform(df[label_column])
        else:
            df['encoded_label'] = self.label_encoder.transform(df[label_column])
        
        logger.info(f"Preprocessing completed. Dataset shape: {df.shape}")
        return df
    
    def create_tfidf_vectorizer(self, texts, max_features=5000, ngram_range=(1, 2)):
        """Create and fit TF-IDF vectorizer"""
        logger.info("Creating TF-IDF vectorizer...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.8,
            token_pattern=r'[^\s]+'  # อนุญาตให้เก็บเครื่องหมายจำเพาะอย่าง __URL__ ได้
        )
        
        # Fit vectorizer
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        return tfidf_matrix
    
    def save_preprocessor(self, vectorizer_path, encoder_path):
        """Save vectorizer and label encoder"""
        if self.vectorizer:
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
        
        if self.label_encoder:
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
        
        logger.info("Preprocessor components saved successfully")
    
    def load_preprocessor(self, vectorizer_path, encoder_path):
        """Load vectorizer and label encoder"""
        try:
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            logger.info("Preprocessor components loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading preprocessor: {e}")
            return False

def load_dataset(file_path):
    """Load dataset from CSV file"""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    preprocessor = ThaiTextPreprocessor()
    
    # โหลดไฟล์ชุดใหม่ที่เราปั๊มข้อมูล Ham เข้าไปแล้วนะครับ
    df = load_dataset("../Dataset/thai_spam_balanced.csv")
    if df is not None:
        # Preprocess
        processed_df = preprocessor.preprocess_dataset(df)
        
        # Create TF-IDF features
        tfidf_matrix = preprocessor.create_tfidf_vectorizer(processed_df['processed_text'])
        
        # Save components
        preprocessor.save_preprocessor(
            "../models/vectorizer.pkl",
            "../models/label_encoder.pkl"
        )
        
        # Save processed data
        processed_df.to_csv("../Dataset/processed_data.csv", index=False)
        logger.info("Processed data saved successfully")