import os
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "lora_model")

# Load dataset for exact match
with open(os.path.join(BASE_DIR, "data", "dataset.json")) as f:
    dataset = json.load(f)

qa_pairs = {item["question"].lower().strip(): item["answer"] for item in dataset}

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    model = PeftModel.from_pretrained(base_model, model_path)
    return tokenizer, model

def answer(question, tokenizer, model):
    key = question.lower().strip()

    # Exact match
    if key in qa_pairs:
        return qa_pairs[key]

    # Word overlap fuzzy match
    question_words = set(key.split())
    best_match = None
    best_score = 0
    for q, a in qa_pairs.items():
        q_words = set(q.split())
        overlap = len(question_words & q_words)
        score = overlap / max(len(question_words), len(q_words))
        if score > best_score:
            best_score = score
            best_match = a

    if best_score >= 0.6:
        return best_match

    # Fallback to model
    inputs = tokenizer(question, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# UI
st.set_page_config(page_title="Deepak's AI Assistant", page_icon="🤖")
st.title("🤖 Deepak's AI Assistant")
st.caption("Powered by LoRA fine-tuned flan-t5")

tokenizer, model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = answer(prompt, tokenizer, model)
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
