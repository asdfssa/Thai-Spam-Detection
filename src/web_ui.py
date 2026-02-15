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
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .spam-box {
        background-color: #ff6b6b;
        color: white;
    }
    .ham-box {
        background-color: #2d6a4f;
        color: white;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-ready { background-color: #2d6a4f; }
    .status-processing { background-color: #ffd43b; }
    .status-error { background-color: #ff6b6b; }
</style>
""", unsafe_allow_html=True)

class ThaiSpamDetectionUI:
    def __init__(self):
        self.model = None
        self.evaluator = None
        self.model_loaded = False
        
    def load_model_components(self):
        """Load model and preprocessing components"""
        try:
            self.model = SpamDetectionModel()
            self.evaluator = ModelEvaluator()
            
            # Try to load model and components
            model_path = "models/spam_detection_model.pkl"
            vectorizer_path = "models/vectorizer.pkl"
            encoder_path = "models/label_encoder.pkl"
            
            if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(encoder_path):
                model_loaded = self.model.load_model(model_path)
                preprocessor_loaded = self.model.preprocessor.load_preprocessor(vectorizer_path, encoder_path)
                
                if model_loaded and preprocessor_loaded:
                    self.model_loaded = True
                    return True
            
            return False
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">🛡️ Thai Spam Detection System</h1>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Render sidebar with model status and information"""
        st.sidebar.markdown("## 📊 System Status")
        
        # Model status
        if self.model_loaded:
            st.sidebar.markdown(
                '<span class="status-indicator status-ready"></span>Model Ready', 
                unsafe_allow_html=True
            )
            if self.model.model_name:
                st.sidebar.info(f"Model: {self.model.model_name}")
        else:
            st.sidebar.markdown(
                '<span class="status-indicator status-error"></span>Model Not Loaded', 
                unsafe_allow_html=True
            )
            st.sidebar.warning("Please train the model first using the training script.")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("## ℹ️ Information")
        st.sidebar.info("""
        This system detects spam messages in Thai language using machine learning.
        
        **Features:**
        - Thai text preprocessing with pythainlp
        - Multiple ML algorithms
        - Real-time prediction
        - Confidence scoring
        - Performance visualization
        """)
    
    def render_prediction_interface(self):
        """Render main prediction interface"""
        st.markdown("## 🔍 Message Analysis")
        
        # Sample messages
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
        
        # --- Callback Function สำหรับจัดการปุ่มกด ---
        def update_text(action):
            import random
            if action == 'spam':
                st.session_state.message_input = random.choice(sample_spam_messages)
            elif action == 'ham':
                st.session_state.message_input = random.choice(sample_ham_messages)
            elif action == 'clear':
                st.session_state.message_input = ""

        # Initialize session state (ถ้ายังไม่มี)
        if 'message_input' not in st.session_state:
            st.session_state.message_input = ""
            
        # Input section
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
            # ใช้ on_click เพื่อสั่งให้ Streamlit ทำงานอัปเดตค่าก่อนวาดหน้าใหม่
            st.button("📋 Sample Spam", use_container_width=True, on_click=update_text, args=('spam',))
            st.button("📋 Sample Ham", use_container_width=True, on_click=update_text, args=('ham',))
            st.button("🗑️ Clear", use_container_width=True, on_click=update_text, args=('clear',))
        
        # Prediction button
        if st.button("🚀 Analyze Message", use_container_width=True, type="primary"):
            if not self.model_loaded:
                st.error("❌ Model not loaded. Please train the model first.")
                return
            
            if not message_text.strip():
                st.warning("⚠️ Please enter a message to analyze.")
                return
            
            # Show processing status
            with st.spinner("🔄 Analyzing message..."):
                time.sleep(1)  # Simulate processing
                
                try:
                    # Make prediction
                    result = self.model.predict(message_text)
                    
                    if result:
                        self.render_prediction_result(result, message_text)
                    else:
                        st.error("❌ Failed to analyze message.")
                        
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
    
    def render_prediction_result(self, result, original_text):
        """Render prediction results with visualization"""
        st.markdown("### 📊 Analysis Results")
        
        # Result box
        if result['label'].lower() == 'spam':
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
        
        # Detailed metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Prediction",
                result['label'].upper(),
                delta=None
            )
        
        with col2:
            st.metric(
                "Confidence",
                f"{result['confidence']:.2%}",
                delta=None
            )
        
        with col3:
            spam_prob = result['spam_probability']
            color = "normal" if spam_prob < 0.5 else "inverse"
            st.metric(
                "Spam Risk",
                f"{spam_prob:.2%}",
                delta=None,
                delta_color=color
            )
        
        # Probability visualization
        st.markdown("### 📈 Probability Distribution")
        
        # Create probability chart
        labels = ['Ham', 'Spam']
        values = [
            1 - result['spam_probability'],
            result['spam_probability']
        ]
        colors = ['#2d6a4f', '#ff6b6b']
        
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
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Text analysis details
        st.markdown("### 🔍 Text Analysis Details")
        
        with st.expander("View preprocessing details"):
            # Show original text
            st.markdown("**Original Text:**")
            st.text(original_text)
            
            # Show cleaned text
            cleaned_text = self.model.preprocessor.clean_text(original_text)
            st.markdown("**Cleaned Text:**")
            st.text(cleaned_text)
            
            # Show tokens
            tokens = self.model.preprocessor.tokenize_thai(cleaned_text)
            st.markdown("**Tokens:**")
            st.text(' | '.join(tokens))
            
            st.markdown(f"**Number of tokens:** {len(tokens)}")
    
    def render_batch_analysis(self):
        """Render batch analysis interface"""
        st.markdown("## 📁 Batch Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with messages:",
            type=['csv'],
            help="CSV should have a 'message' column"
        )
        
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
                                result = self.model.predict(str(message))
                                results.append(result)
                            progress_bar.progress((i + 1) / len(df))
                        
                        # Create results dataframe
                        results_df = pd.DataFrame(results)
                        results_df['message'] = df['message']
                        
                        # Show summary statistics
                        st.markdown("### 📊 Analysis Summary")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            spam_count = (results_df['label'] == 'spam').sum()
                            st.metric("Spam Messages", spam_count)
                        
                        with col2:
                            ham_count = (results_df['label'] == 'ham').sum()
                            st.metric("Legitimate Messages", ham_count)
                        
                        with col3:
                            avg_confidence = results_df['confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                        
                        # Visualization
                        st.markdown("### 📈 Results Visualization")
                        
                        # Pie chart
                        fig_pie = px.pie(
                            results_df,
                            names='label',
                            title='Message Classification Distribution',
                            color='label',
                            color_discrete_map={'spam': '#ff6b6b', 'ham': '#2d6a4f'}
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                        # Confidence distribution
                        fig_hist = px.histogram(
                            results_df,
                            x='confidence',
                            color='label',
                            title='Confidence Score Distribution',
                            nbins=20,
                            color_discrete_map={'spam': '#ff6b6b', 'ham': '#2d6a4f'}
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Download results
                        st.markdown("### 💾 Download Results")
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name="spam_analysis_results.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
    
    def render_model_info(self):
        """Render model information and performance metrics"""
        st.markdown("## 🤖 Model Information")
        
        if not self.model_loaded:
            st.warning("⚠️ Model not loaded. Please train the model first.")
            return
        
        # Model details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model Details")
            st.info(f"""
            **Model Type:** {self.model.model_name or 'Unknown'}
            
            **Preprocessing:**
            - Thai text tokenization (pythainlp)
            - TF-IDF vectorization
            - Text normalization
            """)
        
        with col2:
            st.markdown("### Feature Information")
            if self.model.preprocessor.vectorizer:
                feature_count = self.model.preprocessor.vectorizer.get_feature_names_out().shape[0]
                st.info(f"""
                **Vocabulary Size:** {feature_count:,} features
                
                **Vectorization:** TF-IDF
                
                **Labels:** {list(self.model.preprocessor.label_encoder.classes_)}
                """)
        
        # Performance visualization (if available)
        st.markdown("### 📊 Performance Metrics")
        
        # Check if evaluation results exist
        eval_results_path = "results/evaluation_report.txt"
        if os.path.exists(eval_results_path):
            try:
                with open(eval_results_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                
                with st.expander("View Detailed Evaluation Report"):
                    st.text(report_content)
                
                # Try to load and display confusion matrix
                cm_path = "results/confusion_matrix.png"
                if os.path.exists(cm_path):
                    st.image(cm_path, caption="Confusion Matrix")
                
            except Exception as e:
                st.warning(f"Could not load evaluation results: {e}")
        else:
            st.info("📝 No evaluation results found. Run the evaluation script to generate performance metrics.")
    
    def run(self):
        """Main application runner"""
        # Load model components
        self.load_model_components()
        
        # Render UI components
        self.render_header()
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["🔍 Single Message", "📁 Batch Analysis", "🤖 Model Info"])
        
        with tab1:
            self.render_prediction_interface()
        
        with tab2:
            self.render_batch_analysis()
        
        with tab3:
            self.render_model_info()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666;'>"
            "Thai Spam Detection System © 2024 | Built with Streamlit & pythainlp"
            "</div>", 
            unsafe_allow_html=True
        )

# Main execution
if __name__ == "__main__":
    ui = ThaiSpamDetectionUI()
    ui.run()