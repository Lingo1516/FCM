import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 內建數據 (直接把你的 14x14 矩陣寫在這裡)
# ==========================================
# 這是從你的圖片辨識出來的數據
concepts = [f"C{i+1}" for i in range(14)] # 暫時命名為 C1 到 C14
weights = np.array([
    [0, 0.65, 0.48, 0, 0, -0.4, 0.7, 0, 0, 0, 0, 0.7, 0, 0],       # C1
    [0.7, 0, 0.8, 0, -0.2, -0.73, 0, 0, 0.7, 0, 0.63, 0, -0.3, 0], # C2
    [0, 0.61, 0, 0.7, 0, 0, 0, -0.6, 0, 0, 0.3, 0, -0.4, 0],       # C3
    [0.28, 0, 0, 0, 0, 0, -0.38, 0, 0, 0, 0, 0, 0, 0],             # C4
    [0, -0.68, -0.68, 0, 0, 0, 0, 0.48, 0, 0.6, -0.58, -0.4, 0.33, -0.4], # C5
    [-0.7, -0.73, -0.8, 0, 0, 0, 0, 0, -0.6, 0.4, -0.4, 0, 0.4, 0], # C6
    [0.68, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.43, -0.2, 0],           # C7
    [0, -0.75, -0.6, 0, 0.65, 0.2, 0, 0, 0, 0, 0, -0.3, 0.9, 0],   # C8
    [0.38, 0.6, 0.31, 0, -0.3, 0, 0, 0, 0, 0, 0.43, 0, 0, 0],      # C9
    [0, 0.4, 0, 0, 0, -0.33, 0, 0, 0.4, 0, 0.45, 0, 0, 0.23],      # C10
    [0, 0.41, 0.78, 0, -0.23, 0, 0, 0, 0.7, 0, 0, 0.33, 0, 0.38],  # C11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                    # C12 (看起來全0)
    [-0.35, -0.68, -0.5, 0, 0.73, 0.73, 0, 0.6, 0, 0.3, 0, -0.73, 0, 0], # C13
    [0, 0.25, 0, 0, -0.28, 0, 0, 0, 0, 0, 0, 0.28, 0, 0]           # C14
])

# ==========================================
# 2. 網頁介面設定
# ==========================================
st.set_page_config(page_title="FCM 測試版", layout="wide")
st.title("FCM 模糊認知圖 - 快速測試版")
st.write("此版本已內建 14x14 矩陣數據，無需上傳檔案即可執行。")

# 側邊欄參數
st.sidebar.header("參數設定")
LAMBDA = st.sidebar.slider("Lambda (敏感度)", 0.1, 5.0, 1.0, 0.1)
MAX_STEPS = st.sidebar.slider("模擬次數 (Steps)", 10, 100, 30, 1)
EPSILON = 0.001

# ==========================================
# 3. FCM 運算核心
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
# 4. 執行與顯示
# ==========================================
if st.button('🚀 執行模擬 (Run Simulation)'):
    # 預設初始狀態：假設 C1 (第一個概念) 被觸發，其他為 0
    initial_state = np.zeros(14)
    initial_state[0] = 1.0 
    
    # 執行運算
    results = run_fcm(weights, initial_state, LAMBDA, MAX_STEPS, EPSILON)
    
    # --- 顯示圖表 ---
    st.subheader("趨勢分析圖")
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(14):
        ax.plot(results[:, i], label=f"C{i+1}", alpha=0.7)
    
    ax.set_xlabel("時間 (Steps)")
    ax.set_ylabel("激活程度 (Activation)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)
    
    # --- 顯示數據 ---
    st.subheader("最終穩定狀態數據")
    final_state = results[-1]
    df_res = pd.DataFrame({
        "概念代號": concepts,
        "最終數值": final_state
    }).sort_values(by="最終數值", ascending=False)
    
    st.dataframe(df_res)
    
    # --- 讓你下載整理好的 CSV ---
    st.success("測試成功！如果你需要把這個內建的矩陣下載下來備份，請按下面按鈕：")
    df_export = pd.DataFrame(weights, columns=concepts, index=concepts)
    csv = df_export.to_csv().encode('utf-8')
    st.download_button(
        "📥 下載此矩陣為 CSV",
        csv,
        "fcm_matrix_14x14.csv",
        "text/csv"
    )
