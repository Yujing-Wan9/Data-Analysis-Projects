
---
# Hotel-Insight AI: 基於 RAG 架構的飯店評論分析系統
不再手動翻閱數千則評論！透過 RAG 技術，讓 AI 直接告訴你飯店的優缺點。

在預訂飯店時，消費者常迷失在數千則碎片化的評論中。本專案透過 RAG (Retrieval-Augmented Generation) 架構，讓使用者可以直接詢問飯店相關問題，系統會從評論中找出最相關的內容並生成回答。

# 🛠️ 技術架構
本系統實作了 RAG workflow，將非結構化的評論轉化為可查詢的知識庫：

文本向量化 (Embedding): 使用 all-MiniLM-L6-v2 將 2,000+ 則評論轉換為高維向量。

向量搜尋 (Vector DB): 利用 FAISS 進行極速的 L2 距離相似度檢索。

生成模型 (LLM): 整合 HuggingFace flan-t5-large 進行上下文理解與回答生成。

互動界面: 使用 Gradio 搭建 Web UI，提供直觀的問答體驗。

---
## 資料來源
本專案使用公開的飯店評論資料集，並擷取約 2000 筆評論進行。
https://github.com/bookingcom/ml-dataset-reviews/tree/main/rectour24

每筆資料包含：

- accommodation_id（飯店識別碼）
- review_text（評論內容）
- review_score（評分）

---
## 專案目的

本專案主要用於學習與實作以下 AI 技術：

- Retrieval-Augmented Generation (RAG)
- 向量資料庫應用
- LLM 與文本分析
- AI 系統整合

