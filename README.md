# Data-Analysis & AI-Projects

這個 Repository 收錄了我在大學期間完成的資料分析與 AI 專案，  
涵蓋空間資料分析、機器學習、人臉辨識及RAG等應用。

---
## 1️⃣ Hotel Review RAG 問答系統 (Hotel Review QA System with RAG)

專案簡介：
使用 Python 建立簡易 RAG (Retrieval-Augmented Generation) 系統，
透過飯店評論資料建立向量檢索機制，並結合語言模型生成回答。
系統可根據使用者問題搜尋相關評論內容並生成摘要回覆。

使用工具：
Python、Sentence Transformers、FAISS、HuggingFace Transformers、Gradio

成果檔案：
```
├── hotelreview_Demo.py  
├── API.py  
├── hotel_reviews_2000.csv  
├── requirements.txt  
└── README.md
```

## 2️⃣ 核電廠選址分析 (Nuclear Power Plant Site Selection)

**專案簡介：**  
使用 GIS 與 Python (Pandas) 進行空間資料分析，整合人口密度、火山距離與震源深度等地理資料，  
透過多重準則決策分析 (MCDA) 評估核電廠適合地點。

**使用工具：**  
ArcGIS Pro、Python (Pandas, Matplotlib)、Excel

**成果檔案：**  
```
├── k_means_clustering_forNuclear.ipynb
├── 核電廠選址與災害風險分析report.pdf
├──Nuclear_spatialData.ppkx 
└── README.md
```

---

## 3️⃣ YouTuber 潛力預測 (YouTuber Growth Prediction)

**專案簡介：**  
使用 Python 建立機器學習模型，預測 YouTuber 未來成長潛力。  
分析頻道觀看數、訂閱數、上傳頻率等特徵，進行資料清理、特徵選取與回歸模型訓練，最終可視化結果。

**使用工具：**  
Python (Pandas, Scikit-learn)、Colab Notebook

**成果檔案：**  
```
├── Model_performance.xlsx
├── Decision_index.ipynb
├── Youtuber_Growth_Prediction.docx
├── Youtuber_Growth_Prediction_Report.pdf
└── read.txt
```

---

## 4️⃣ 人臉辨識專案－再犯車手遮蔽辨識 (Masked Face Recognition)

**專案簡介：**  
建立人臉辨識系統，針對遮蔽物（口罩、帽子等）辨識，  
利用Insightface和Catboost模型訓練影像資料，測試在不同遮蔽比例下的辨識準確率。

**使用工具：**  
Python、Colab Notebook

**成果檔案：**  
人臉辨識專案－再犯車手遮蔽辨識
```
├── demo.py
├── output.csv
├── README.md
└── requirements.txt
```
---

## 5️⃣ 茶園適合地區分析 (Tea Plantation Suitability)

**專案簡介：**  
分析台灣石碇區永安里茶園種植適合性，Slope 和NDVI 數值探討
**使用工具：**  
ArcGIS pro、Raster Data

**成果檔案：**  
```
├── 永安里空間分析.aprx
├── 石碇區和茶園植被評估-report.pptx
└── README.md
```
