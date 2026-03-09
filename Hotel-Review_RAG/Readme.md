
---
# 建立一個利用 AI 分析飯店評論的系統。  
使用 Retrieval-Augmented Generation (RAG) 架構，讓使用者可以直接詢問飯店相關問題，系統會從評論中找出最相關的內容並生成回答。

##本專案使用以下技術：
- Python
- Sentence Transformers（文本向量化）
- FAISS（向量搜尋）
- HuggingFace Transformers（LLM）
- Gradio（互動式 Demo）
- Pandas / NumPy（資料處理）

---
## 資料來源
本專案使用公開的飯店評論資料集，並擷取約 2000 筆評論進行實驗與系統開發。
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

