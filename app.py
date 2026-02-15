import streamlit as st
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the main web UI
from web_ui import ThaiSpamDetectionUI

if __name__ == "__main__":
    ui = ThaiSpamDetectionUI()
    ui.run()
