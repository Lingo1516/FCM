
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. 網頁標題與設定
st.title("FCM 模糊認知圖模擬器")
st.write("這是一個互動式的 FCM 模擬工具，你可以調整參數並觀察系統收斂過程。")

# 2. 側邊欄：讓使用者調整參數
st.sidebar.header("參數設定 (Configuration)")
LAMBDA = st.sidebar.slider("Lambda (敏感度)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
MAX_STEPS = st.sidebar.slider("最大疊代次數", min_value=10, max_value=100, value=50, step=5)
EPSILON = 0.001

# 3. 定義數據 (這裡可以未來擴充成讓使用者上傳 CSV)
concepts = ["C1 文化性商品", "C2 競爭對手能力", "C3 定期會議", "C4 配銷量", "C5 供應鏈信賴"]

# 權重矩陣 (這裡用範例，你可以換成你原本的 13x13)
weights = np.array([
    [0.0,  0.6, -0.2,  0.0,  0.5],
    [0.0,  0.0,  0.4, -0.5,  0.0],
    [0.3,  0.0,  0.0,  0.7,  0.8],
    [0.0,  0.0,  0.0,  0.0,  0.4],
    [0.2, -0.3,  0.0,  0.0,  0.0]
])

# 讓使用者選擇初始狀態
st.sidebar.subheader("初始狀態設定")
init_c3 = st.sidebar.slider("C3 定期會議 初始投入量", 0.0, 1.0, 1.0)
initial_state = np.array([0, 0, init_c3, 0, 0])

# 4. 運算邏輯
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

# 執行按鈕
if st.button('開始模擬'):
    results = run_fcm(weights, initial_state, LAMBDA, MAX_STEPS, EPSILON)
    
    # 5. 顯示結果圖表
    st.subheader("模擬趨勢圖")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 設定中文字型 (Streamlit Cloud 預設不支援中文，這裡用英文或通用字型避免亂碼，進階需另外設定)
    # 為了避免報錯，我們先暫時用英文 Legend
    eng_concepts = ["C1", "C2", "C3", "C4", "C5"]
    
    for i in range(len(concepts)):
        ax.plot(results[:, i], label=f"{concepts[i]}", marker='o', markersize=3)
    
    ax.set_title(f'FCM Simulation (Lambda={LAMBDA})')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Activation Value')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 關鍵：用 st.pyplot 顯示，而不是 plt.show()
    st.pyplot(fig)
    
    # 顯示數據表格
    st.subheader("詳細數據")
    df = pd.DataFrame(results, columns=concepts)
    st.dataframe(df)
