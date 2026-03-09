# -*- coding: utf-8 -*-
import pandas as pd
import faiss
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. 載入資料 (改為相對路徑)
print("Loading data...")
try:
    df = pd.read_csv("hotel_reviews_2000.csv", encoding="utf-8")
    reviews = df["review_text"].tolist()
except FileNotFoundError:
    print("Error: 找不到 hotel_reviews_2000.csv，請確保檔案在同一個資料夾。")
    reviews = []

# 2. 初始化模型 (放在全域，避免每次 query 都重新載入)
print("Initializing models...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(reviews)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def ask_hotel(question, k=5):
    # 找相關評論
    query_vector = embed_model.encode([question])
    distances, indices = index.search(np.array(query_vector).astype('float32'), k)
    
    retrieved_reviews = [reviews[i] for i in indices[0]]
    context = "\n".join(retrieved_reviews)

    # Prompt
    prompt = f"""You are a hotel review assistant.
Based on the following hotel reviews:
{context}

Answer the question: {question}

Also provide:
Pros:
Cons:
"""

    # LLM 生成
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = llm_model.generate(**inputs, max_new_tokens=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer

# 3. Gradio 介面
def hotel_demo(question):
    return ask_hotel(question)

demo = gr.Interface(
    fn=hotel_demo,
    inputs=gr.Textbox(
        label="Ask about the hotel",
        placeholder="Example: Is the hotel quiet?"
    ),
    outputs=gr.Textbox(label="AI Answer"),
    title="Hotel Review AI Assistant",
    description="Ask questions about hotel reviews using AI."
)

if __name__ == "__main__":
    # 啟動 Gradio，加上 share=True 可以產生臨時連結給別人看
    demo.launch()
