import gradio as gr
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from insightface.app import FaceAnalysis
from PIL import Image
import cv2
import os

# ==========================
# 1️⃣ 讀取訓練資料
# ==========================

print("Loading training data...")

DATA_PATH = "output.csv"

df_train = pd.read_csv(DATA_PATH)

feature_cols = [c for c in df_train.columns if c not in ['測試照片','Baseline照片','結果','label']]

X_train = df_train[feature_cols]
y_train = df_train['label']

# ==========================
# 2️⃣ 訓練 CatBoost 模型
# ==========================

print("Training CatBoost model...")

model = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.1,
    loss_function='Logloss',
    eval_metric='AUC',
    verbose=False
)

model.fit(Pool(X_train, y_train))

print("CatBoost training completed")

# ==========================
# 3️⃣ InsightFace 初始化
# ==========================

print("Loading InsightFace model...")

app = FaceAnalysis()
app.prepare(ctx_id=0)

# ==========================
# 4️⃣ 計算人臉相似度
# ==========================

def compute_similarity(img1_path, img2_path):

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    faces1 = app.get(img1)
    faces2 = app.get(img2)

    if len(faces1) == 0 or len(faces2) == 0:
        return 0.0

    feat1 = faces1[0].embedding
    feat2 = faces2[0].embedding

    cos_sim = np.dot(feat1, feat2) / (
        np.linalg.norm(feat1) * np.linalg.norm(feat2)
    )

    return float(cos_sim)

# ==========================
# 5️⃣ Gradio 預測函式
# ==========================

def predict_demo(baseline_img, test_img, glasses, mask, hat, helmet):

    baseline_img_path = "baseline.jpg"
    test_img_path = "test.jpg"

    baseline_img.save(baseline_img_path)
    test_img.save(test_img_path)

    sim = compute_similarity(baseline_img_path, test_img_path)

    manual_features = {
        'glasses': int(glasses),
        'mask': int(mask),
        'hat': int(hat),
        'Lhelmet': int(helmet)
    }

    combined = {c: 0 for c in feature_cols}
    combined.update(manual_features)
    combined['相似度'] = sim

    X_test = pd.DataFrame([combined], columns=feature_cols)

    proba = model.predict_proba(X_test)[:,1][0]

    label = "✅ Same person" if proba > 0.5 else "❌ Not same person"

    return f"""
Prediction: {label}

Face Similarity: {sim:.3f}

CatBoost Probability: {proba:.3f}
"""

# ==========================
# 6️⃣ Gradio 介面
# ==========================

iface = gr.Interface(
    fn=predict_demo,
    inputs=[
        gr.Image(type="pil", label="Baseline Photo"),
        gr.Image(type="pil", label="Test Photo"),
        gr.Checkbox(label="Glasses 👓"),
        gr.Checkbox(label="Mask 😷"),
        gr.Checkbox(label="Hat 🧢"),
        gr.Checkbox(label="Helmet 🪖")
    ],
    outputs=gr.Textbox(label="Prediction Result", lines=6),
    title="Face Recognition Demo",
    description="Upload two face images and select accessories to evaluate identity similarity."
)

# ==========================
# 7️⃣ 啟動 Demo
# ==========================

if __name__ == "__main__":
    iface.launch()
