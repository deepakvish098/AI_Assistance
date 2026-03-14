# import os
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from peft import PeftModel
# import json

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# model_path = os.path.join(BASE_DIR, "models", "lora_model")

# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
# tokenizer.pad_token = tokenizer.eos_token

# base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
# model = PeftModel.from_pretrained(base_model, model_path)

# input_text = "Question:who build you?\nAnswer:"
# inputs = tokenizer(input_text, return_tensors="pt")

# # output = model.generate(
# #     **inputs,
# #     max_new_tokens=100,
# #     do_sample=False,
# #     temperature=0.7,
# #     repetition_penalty=1.3,
# # )
# output = model.generate(**inputs, max_length=50)



# print(tokenizer.decode(output[0], skip_special_tokens=True))

import os
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load dataset for exact match lookup
with open(os.path.join(BASE_DIR, "data", "dataset.json")) as f:
    dataset = json.load(f)

qa_pairs = {item["question"].lower().strip(): item["answer"] for item in dataset}

# Load model as fallback
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
model_path = os.path.join(BASE_DIR, "models", "lora_model")
model = PeftModel.from_pretrained(base_model, model_path)

# def answer(question):
#     # First try exact match from dataset
#     key = question.lower().strip()
#     if key in qa_pairs:
#         return qa_pairs[key]
    
#     # Fallback to model
#     inputs = tokenizer(question, return_tensors="pt")
#     output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
#     return tokenizer.decode(output[0], skip_special_tokens=True)

def answer(question):
    key = question.lower().strip()
    
    # Exact match
    if key in qa_pairs:
        return qa_pairs[key]
    
    # Fuzzy match using word overlap
    question_words = set(key.split())
    best_match = None
    best_score = 0

    for q, a in qa_pairs.items():
        q_words = set(q.split())
        overlap = len(question_words & q_words)  # count common words
        score = overlap / max(len(question_words), len(q_words))
        if score > best_score:
            best_score = score
            best_match = a

    if best_score >= 0.5:  # at least 50% word overlap
        return best_match
    
    # Fallback to model
    inputs = tokenizer(question, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Test
print(answer("who built you?"))
print(answer("What is AI?"))
print(answer("What is the capital of France?"))
