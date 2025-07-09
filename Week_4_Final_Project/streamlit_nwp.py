import streamlit as st
from transformers import GPT2LMHeadModel, AutoTokenizer
import os

@st.cache_resource
def load_model():
    # Use absolute path to avoid any path resolution issues
    model_path = os.path.abspath("./Week_4_Final_Project/gpt2-finetuned-nwp-final")

    # Load with explicit local_files_only=True
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, device_map="cpu")
        model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True, device_map="cpu")
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        # List directory contents for debugging
        if os.path.exists(model_path):
            st.write("Directory contents:", os.listdir(model_path))
        else:
            st.write(f"Directory does not exist: {model_path}")
        return None, None

model, tokenizer = load_model()