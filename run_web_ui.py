"""
Web UI launcher for Thai Spam Detection System
Run this script to start the Streamlit web interface
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit web UI"""
    print("=" * 60)
    print("THAI SPAM DETECTION WEB UI")
    print("=" * 60)
    
    # Check if model exists
    model_path = "models/spam_detection_model.pkl"
    if not os.path.exists(model_path):
        print("⚠️  Warning: Model not found!")
        print("Please run 'python train_model.py' first to train the model.")
        print("The web UI will still launch but prediction features will be limited.")
        print()
    
    # Launch Streamlit
    try:
        print("🚀 Starting Streamlit web UI...")
        print("📱 The UI will open in your default browser")
        print("🌐 Local URL: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/web_ui.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Web UI stopped by user")
    except Exception as e:
        print(f"❌ Error starting web UI: {e}")
        print("Make sure streamlit is installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
