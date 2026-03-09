## 如何執行 Demo

請依照以下步驟執行本專案的 Demo 系統。

### 1️⃣ 下載專案

先使用 git clone 下載本專案：git clone https://github.com/Yujing-Wan9/Data-Analysis-Projects.git

### 2️⃣ 進入專案資料夾
cd Data-Analysis-Projects/人臉辨識專案－再犯車手遮蔽辨識
### 3️⃣ 安裝必要套件
請先安裝專案所需的 Python 套件：pip install -r requirements.txt
### 4️⃣ 執行 Demo 系統
執行以下指令啟動 Gradio Demo：
python demo.py
### 5️⃣ 開啟瀏覽器
程式啟動後，終端機會顯示：
Running on http://127.0.0.1:7860
請在瀏覽器開啟

即可使用人臉辨識 Demo 系統。

---

## Demo 使用方式

1️⃣ 上傳 **Baseline 照片**（基準照片）  
2️⃣ 上傳 **Test 照片**（測試照片）  
3️⃣ 勾選人物配件（眼鏡、口罩、帽子、安全帽）  
4️⃣ 系統會計算：

- 人臉相似度
- CatBoost 預測機率
- 是否為同一人

最後顯示預測結果。




