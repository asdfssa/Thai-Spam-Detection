"""
Streamlit Web UI for Thai Spam Detection
Provides user-friendly interface for spam detection with visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_training import SpamDetectionModel
from model_evaluation import ModelEvaluator

# --- โหลดโมเดล Deep Learning จาก Hugging Face (สำหรับ Cloud Deployment) ---
@st.cache_resource
def load_hf_pipeline():
    try:
        from transformers import pipeline
        model_id = "MuneTH1/thai-spam-wangchanberta" 
        
        return pipeline("text-classification", model=model_id, tokenizer=model_id)
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {e}")
        return None

# Page configuration
st.set_page_config(
    page_title="Thai Spam Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4dabf7; 
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .spam-box {
        background-color: rgba(255, 107, 107, 0.2); 
        border: 1px solid #ff6b6b;
        color: #ffc9c9;
    }
    .ham-box {
        background-color: rgba(45, 106, 79, 0.4); 
        border: 1px solid #40c057;
        color: #b2f2bb;
    }
    /* 🔥 เพิ่มสีเหลืองสำหรับความมั่นใจต่ำ */
    .suspicious-box {
        background-color: rgba(255, 212, 59, 0.2); 
        border: 1px solid #ffd43b;
        color: #fff3bf;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-ready { background-color: #40c057; }
    .status-processing { background-color: #ffd43b; }
    .status-error { background-color: #ff6b6b; }
</style>
""", unsafe_allow_html=True)

class ThaiSpamDetectionUI:
    def __init__(self):
        self.model = None
        self.hf_model = None  
        self.evaluator = None
        self.model_loaded = False
        self.active_model_type = "Deep Learning (WangchanBERTa)" 
        
    def load_model_components(self):
        """Load model and preprocessing components"""
        try:
            # 1. โหลด TF-IDF แบบเดิม
            self.model = SpamDetectionModel()
            self.evaluator = ModelEvaluator()
            
            model_path = "models/spam_detection_model.pkl"
            vectorizer_path = "models/vectorizer.pkl"
            encoder_path = "models/label_encoder.pkl"
            
            if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(encoder_path):
                self.model.load_model(model_path)
                self.model.preprocessor.load_preprocessor(vectorizer_path, encoder_path)
            
            # 2. โหลด Deep Learning
            self.hf_model = load_hf_pipeline()
            
        except Exception as e:
            st.error(f"Error loading models: {e}")

    def predict_message(self, text):
        if self.active_model_type == "Machine Learning (TF-IDF)":
            return self.model.predict(text)
        elif self.active_model_type == "Deep Learning (WangchanBERTa)":
            if not self.hf_model:
                return None
            result = self.hf_model(text)[0]
            is_spam = result['label'] == 'LABEL_1'
            confidence = result['score']
            return {
                'label': 'spam' if is_spam else 'ham',
                'confidence': confidence,
                'spam_probability': confidence if is_spam else 1 - confidence
            }
        return None
    
    def render_header(self):
        st.markdown('<h1 class="main-header">🛡️ Thai Spam Detection System</h1>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        st.sidebar.markdown("## ⚙️ Model Selection")
        self.active_model_type = st.sidebar.selectbox(
            "Choose Detection Model:",
            ["Deep Learning (WangchanBERTa)", "Machine Learning (TF-IDF)"]
        )
        st.sidebar.markdown("---")
        
        if self.active_model_type == "Machine Learning (TF-IDF)":
            self.model_loaded = (self.model is not None and getattr(self.model, 'model', None) is not None)
        else:
            self.model_loaded = (self.hf_model is not None)

        st.sidebar.markdown("## 📊 System Status")
        
        if self.model_loaded:
            st.sidebar.markdown(
                '<span class="status-indicator status-ready"></span>Model Ready', 
                unsafe_allow_html=True
            )
            st.sidebar.info(f"Active: {self.active_model_type.split(' ')[0]}")
        else:
            st.sidebar.markdown(
                '<span class="status-indicator status-error"></span>Model Not Loaded', 
                unsafe_allow_html=True
            )
            st.sidebar.warning("Model files missing for selected type.")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("## ℹ️ Information")
        st.sidebar.info("""
        This system detects spam messages in Thai language using machine learning.
        
        **Features:**
        - Thai text preprocessing with pythainlp
        - Multiple ML algorithms (TF-IDF & Deep Learning)
        - Real-time prediction
        - Confidence scoring (Yellow Alert < 0.70)
        - Performance visualization
        """)
    
    def render_prediction_interface(self):
        st.markdown(f"## 🔍 Message Analysis *(Using: {self.active_model_type.split(' ')[0]})*")
        
        sample_spam_messages = [
            "มีคดีค้างชำระ ตรวจสอบด่วน http://bank-confirm.site/09wq0o",
            "ระบบพบความผิดปกติ OTP:9558 กรอกที่ http://delivery-check.me/o15ls9 💰",
            "เรื่องที่คุยไว้ ลองเข้าไปดู http://promo-sale99.net/39usl5 ก่อน 💸",
            "โอน 50175 บาทแล้วนะ ดูสลิปที่ http://delivery-check.me/f89ztm",
            "คุณชนะรางวัล! รับของรางวัลที่ http://winner-claim.net/abc123"
        ]
        
        sample_ham_messages = [
            "ประชุม 19 โมงนะ",
            "ส่งไฟล์ให้แล้ว http://youtube.com/4tduad",
            "ร้านนี้ลด 50% จริง 555",
            "เอ โอนเงินให้แล้ว 19996 บาทนะ 🚨",
            "ดูคลิปนี้สิ http://facebook.com/k0ai5i 😱"
        ]
        
        def update_text(action):
            import random
            if action == 'spam':
                st.session_state.message_input = random.choice(sample_spam_messages)
            elif action == 'ham':
                st.session_state.message_input = random.choice(sample_ham_messages)
            elif action == 'clear':
                st.session_state.message_input = ""

        if 'message_input' not in st.session_state:
            st.session_state.message_input = ""
            
        col1, col2 = st.columns([3, 1])
        
        with col1:
            message_text = st.text_area(
                "Enter Thai message to analyze:",
                placeholder="พิมพ์ข้อความภาษาไทยที่ต้องการตรวจสอบ...",
                height=150,
                key="message_input"
            )
        
        with col2:
            st.markdown("### Quick Actions")
            st.button("📋 Sample Spam", use_container_width=True, on_click=update_text, args=('spam',))
            st.button("📋 Sample Ham", use_container_width=True, on_click=update_text, args=('ham',))
            st.button("🗑️ Clear", use_container_width=True, on_click=update_text, args=('clear',))
        
        if st.button("🚀 Analyze Message", use_container_width=True, type="primary"):
            if not self.model_loaded:
                st.error("❌ Model not loaded. Please train or provide the model first.")
                return
            
            if not message_text.strip():
                st.warning("⚠️ Please enter a message to analyze.")
                return
            
            with st.spinner(f"🔄 Analyzing message with {self.active_model_type.split(' ')[0]}..."):
                time.sleep(1) 
                
                try:
                    result = self.predict_message(message_text)
                    
                    if result:
                        self.render_prediction_result(result, message_text)
                    else:
                        st.error("❌ Failed to analyze message.")
                        
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
    
    def render_prediction_result(self, result, original_text):
        st.markdown("### 📊 Analysis Results")
        
        # 🔥 Logic ระบบ 3 สี (เขียว/เหลือง/แดง)
        if result['confidence'] < 0.70:
            st.markdown(f"""
            <div class="prediction-box suspicious-box">
                <h3>🟡 น่าสงสัย (กรุณาตรวจสอบซ้ำ)</h3>
                <p><strong>Predicted as:</strong> {result['label'].upper()}</p>
                <p><strong>Confidence:</strong> {result['confidence']:.2%} (ต่ำกว่าเกณฑ์ 70.00%)</p>
                <p><strong>Spam Probability:</strong> {result['spam_probability']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        elif result['label'].lower() == 'spam':
            st.markdown(f"""
            <div class="prediction-box spam-box">
                <h3>🚨 SPAM DETECTED!</h3>
                <p><strong>Confidence:</strong> {result['confidence']:.2%}</p>
                <p><strong>Spam Probability:</strong> {result['spam_probability']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box ham-box">
                <h3>✅ legitimate message</h3>
                <p><strong>Confidence:</strong> {result['confidence']:.2%}</p>
                <p><strong>Spam Probability:</strong> {result['spam_probability']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prediction", result['label'].upper())
        
        with col2:
            st.metric("Confidence", f"{result['confidence']:.2%}")
        
        with col3:
            spam_prob = result['spam_probability']
            color = "normal" if spam_prob < 0.5 else "inverse"
            st.metric("Spam Risk", f"{spam_prob:.2%}", delta_color=color)
        
        st.markdown("### 📈 Probability Distribution")
        
        labels = ['Ham', 'Spam']
        values = [1 - result['spam_probability'], result['spam_probability']]
        colors = ['#40c057', '#ff6b6b'] 
        
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f'{v:.2%}' for v in values],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Classification Probability",
            yaxis_title="Probability",
            showlegend=False,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', 
            font=dict(color='#FAFAFA')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🔍 Text Analysis Details")
        
        with st.expander("View preprocessing details"):
            st.markdown("**Original Text:**")
            st.text(original_text)
            
            if self.active_model_type == "Machine Learning (TF-IDF)":
                cleaned_text = self.model.preprocessor.clean_text(original_text)
                st.markdown("**Cleaned Text:**")
                st.text(cleaned_text)
                
                tokens = self.model.preprocessor.tokenize_thai(cleaned_text)
                st.markdown("**Tokens:**")
                st.text(' | '.join(tokens))
                st.markdown(f"**Number of tokens:** {len(tokens)}")
            else:
                st.markdown("**Processing Method:**")
                st.text("Subword Tokenization (Processed natively by Hugging Face Transformer)")
    
    def render_batch_analysis(self):
        st.markdown("## 📁 Batch Analysis")
        uploaded_file = st.file_uploader("Upload CSV file with messages:", type=['csv'], help="CSV should have a 'message' column")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File loaded successfully! Found {len(df)} messages.")
                
                if st.button("🚀 Analyze All Messages", type="primary"):
                    if not self.model_loaded:
                        st.error("❌ Model not loaded. Please train the model first.")
                        return
                    
                    with st.spinner("🔄 Analyzing messages..."):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, message in enumerate(df['message']):
                            if pd.notna(message):
                                result = self.predict_message(str(message))
                                results.append(result)
                            progress_bar.progress((i + 1) / len(df))
                        
                        results_df = pd.DataFrame(results)
                        results_df['message'] = df['message']
                        
                        st.markdown("### 📊 Analysis Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1: st.metric("Spam Messages", (results_df['label'] == 'spam').sum())
                        with col2: st.metric("Legitimate Messages", (results_df['label'] == 'ham').sum())
                        with col3: st.metric("Avg Confidence", f"{results_df['confidence'].mean():.2%}")
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(label="📥 Download Results CSV", data=csv, file_name="spam_analysis_results.csv", mime="text/csv")
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
    
    def render_model_info(self):
        st.markdown("## 🤖 Model Information")
        if not self.model_loaded:
            st.warning("⚠️ Model not loaded.")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Model Details")
            if self.active_model_type == "Machine Learning (TF-IDF)":
                st.info(f"**Model Type:** {self.model.model_name}\n\n**Preprocessing:**\n- Thai text tokenization\n- TF-IDF vectorization")
            else:
                st.info("**Model Type:** WangchanBERTa (Deep Learning)\n\n**Preprocessing:**\n- Subword Tokenization\n- Context Analysis")
        
        with col2:
            st.markdown("### Feature Information")
            if self.active_model_type == "Machine Learning (TF-IDF)":
                if getattr(self.model.preprocessor, 'vectorizer', None):
                    st.info(f"**Vocabulary Size:** {self.model.preprocessor.vectorizer.get_feature_names_out().shape[0]:,} features")
            else:
                st.info("**Architecture:** RoBERTa-base\n**Parameters:** ~110 Million")

        if self.active_model_type == "Machine Learning (TF-IDF)":
            st.markdown("### 📊 Performance Metrics (TF-IDF)")
            if os.path.exists("results/evaluation_report.txt"):
                with st.expander("View Report"):
                    with open("results/evaluation_report.txt", 'r', encoding='utf-8') as f:
                        st.text(f.read())
    
    def run(self):
        self.load_model_components()
        self.render_header()
        self.render_sidebar()
        tab1, tab2, tab3 = st.tabs(["🔍 Single Message", "📁 Batch Analysis", "🤖 Model Info"])
        with tab1: self.render_prediction_interface()
        with tab2: self.render_batch_analysis()
        with tab3: self.render_model_info()
        st.markdown("---")
        st.markdown("<div style='text-align: center; color: #888;'>Thai Spam Detection System © 2026 | Built with Streamlit & Transformers</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    ui = ThaiSpamDetectionUI()
    ui.run()