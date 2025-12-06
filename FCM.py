import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 頁面基本設定
# ==========================================
st.set_page_config(page_title="FCM 分析工具", layout="wide")
st.title("FCM 模糊認知圖分析器")
st.markdown("""
### 使用說明
1. 請上傳包含權重矩陣的 **Excel (.xlsx)** 或 **CSV** 檔案。
2. 檔案格式：**第一列**與**第一欄**必須是概念名稱 (Concepts)。
""")

# ==========================================
# 2. 側邊欄：參數設定
# ==========================================
st.sidebar.header("1. 參數設定 (Settings)")
LAMBDA = st.sidebar.slider("Lambda (敏感度)", 0.1, 5.0, 1.0, 0.1)
MAX_STEPS = st.sidebar.slider("最大疊代次數", 10, 100, 50, 5)
EPSILON = 0.001

# ==========================================
# 3. 檔案上傳區 (核心修改處)
# ==========================================
st.sidebar.header("2. 資料上傳")
uploaded_file = st.sidebar.file_uploader("上傳矩陣檔", type=['xlsx', 'csv'])

# 預設變數 (如果沒上傳檔案時用的防呆機制)
df = None
weights = None
concepts = []

if uploaded_file is not None:
    try:
        # 判斷是 Excel 還是 CSV
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, index_col=0)
        else:
            df = pd.read_excel(uploaded_file, index_col=0)
            
        # 抓取資料
        concepts = df.columns.tolist()  # 抓取概念名稱
        weights = df.values             # 抓取數值矩陣
        
        st.success(f"成功讀取檔案！偵測到 {len(concepts)} 個概念。")
        
        # 顯示讀取到的矩陣給使用者確認
        with st.expander("點擊查看原始矩陣數據"):
            st.dataframe(df)
            
    except Exception as e:
        st.error(f"檔案讀取錯誤：{e}")
else:
    st.info("👈 請在左側上傳檔案以開始分析")

# ==========================================
# 4. FCM 運算引擎
# ==========================================
def sigmoid(x, lambd):
    return 1 / (1 + np.exp(-lambd * x))

def run_fcm(W, A_init, lambd, steps, epsilon):
    history = [A_init]
    current_state = A_init
    
    for _ in range(steps):
        influence = np.dot(current_state, W)
        next_state = sigmoid(influence, lambd)
        history.append(next_state)
        
        if np.max(np.abs(next_state - current_state)) < epsilon:
            break
        current_state = next_state
        
    return np.array(history)

# ==========================================
# 5. 執行模擬與繪圖
# ==========================================
if df is not None:
    st.sidebar.header("3. 情境模擬")
    
    # 讓使用者設定每個概念的初始值 (自動產生拉桿)
    st.sidebar.subheader("設定初始狀態 (0~1)")
    initial_values = []
    
    # 使用 Form 避免每次拉動都重新整理
    with st.sidebar.form("init_form"):
        for concept in concepts:
            val = st.slider(f"{concept}", 0.0, 1.0, 0.5) # 預設0.5
            initial_values.append(val)
        submitted = st.form_submit_button("開始運算 (Run)")

    if submitted:
        initial_state = np.array(initial_values)
        
        # 執行運算
        results = run_fcm(weights, initial_state, LAMBDA, MAX_STEPS, EPSILON)
        
        # 建立兩欄版面
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("趨勢圖 (Trends)")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 繪圖
            for i, concept in enumerate(concepts):
                ax.plot(results[:, i], label=concept, marker='o', markersize=4)
            
            ax.set_xlabel("Steps (Time)")
            ax.set_ylabel("Activation Level")
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # 處理圖例位置，避免擋住線
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            st.pyplot(fig)

        with col2:
            st.subheader("最終穩定狀態")
            final_state = results[-1]
            # 製作結果表格並排序
            res_df = pd.DataFrame({
                "概念 (Concept)": concepts,
                "最終值 (Value)": final_state
            }).sort_values(by="最終值 (Value)", ascending=False)
            
            st.dataframe(res_df.style.background_gradient(cmap='Blues'), height=400)

        # 下載結果功能
        st.subheader("下載分析結果")
        result_csv = pd.DataFrame(results, columns=concepts).to_csv(index=False).encode('utf-8')
        st.download_button(
            "下載詳細數據 (CSV)",
            result_csv,
            "fcm_results.csv",
            "text/csv"
        )
