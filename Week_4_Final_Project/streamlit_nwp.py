import streamlit as st
from transformers import GPT2LMHeadModel, AutoTokenizer
import torch

# Loading the model and tokenizer
@st.cache_resource
def load_model():
    model_path = "./Week_4_Final_Project/gpt2-finetuned-nwp-final"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# Set page config
st.set_page_config(page_title="GPT-2 Finetuned Generator", layout="wide")
st.markdown("<h1 style='text-align: center;'>GPT-2 Finetuned Text Generator</h1>", unsafe_allow_html=True)
st.markdown("---")

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar for history
with st.sidebar:
    st.markdown("## History")
    if st.button("Clear History"):
        st.session_state.history = []
    if st.session_state.history:
        for i, (p, r) in enumerate(reversed(st.session_state.history[-5:]), 1):
            st.markdown(f"**{i}. Prompt:** {p}")
            st.markdown(f"`→ {r}`")
            st.markdown("---")
    else:
        st.markdown("No generations yet.")

# Two-column layout
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Input Prompt")
    prompt = st.text_area("Enter your prompt:", height=150)

    st.subheader("Generation Settings")
    max_new_tokens = st.slider("Max New Tokens", min_value=1, max_value=5, value=2)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.9, step=0.05)

    generate_btn = st.button("Generate")

with right_col:
    st.subheader("Generated Output")

    if generate_btn:
        if prompt.strip() == "":
            st.warning("Please enter a prompt.")
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id
                )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True, errors="replace")

            st.markdown(f"**Generated:**\n\n{generated_text}")
            st.session_state.history.append((prompt, generated_text))