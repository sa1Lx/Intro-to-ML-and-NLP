import streamlit as st
from transformers import GPT2LMHeadModel, AutoTokenizer
import torch
import os

@st.cache_resource
def load_model():
    model_path = os.path.abspath("./gpt2-finetuned-nwp-final")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()
