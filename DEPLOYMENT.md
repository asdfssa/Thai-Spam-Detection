# Thai Spam Detection System

## 📋 รายละเอียดโปรเจค
โปรเจคตรวจจับสแปมภาษาไทยโดยใช้ Machine Learning กับ Streamlit Web UI

## 🚀 การติดตั้ง
```bash
pip install -r requirements.txt
```

## 🏃 การรันโปรแกรม
### แบบ Local:
```bash
streamlit run app.py
```

### แบบ Web UI Launcher:
```bash
python run_web_ui.py
```

## 📁 โครงสร้างไฟล์
```
Project/
├── app.py                 # Main entry point for Streamlit Cloud
├── run_web_ui.py         # Local web UI launcher
├── requirements.txt      # Python dependencies
├── src/
│   ├── web_ui.py         # Streamlit web interface
│   ├── model_training.py # Model training and prediction
│   ├── data_preprocessing.py # Text preprocessing
│   └── model_evaluation.py # Model evaluation utilities
├── models/
│   ├── spam_detection_model.pkl
│   ├── vectorizer.pkl
│   └── label_encoder.pkl
├── Dataset/
│   └── thai_spam_production_v4.csv
└── results/
    └── confusion_matrix.png
```

## 🤖 คุณสมบัติ
- การประมวลผลข้อความภาษาไทยด้วย pythainlp
- หลายอัลกอริทึม ML (RandomForest, NaiveBayes, SVM, LogisticRegression)
- Real-time prediction พร้อม confidence score
- Batch analysis สำหรับหลายข้อความ
- Visualization ของผลลัพธ์
- TF-IDF vectorization

## 🌐 Deployment
โปรเจคนี้พร้อมสำหรับการ deploy บน Streamlit Cloud
1. Push code ขึ้น GitHub repository
2. เชื่อมต่อกับ Streamlit Cloud
3. ระบุ `app.py` เป็น main file

## 📊 Performance
- ใช้ TF-IDF features สูงสุด 5000 คำ
- Cross-validation 5-fold
- Hyperparameter tuning ด้วย GridSearchCV
- ประเมินผลด้วย confusion matrix และ classification report
