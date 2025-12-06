import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 頁面基本設定
# ==========================================
st.set_page_config(page_title="FCM 策略模擬系統", layout="wide")
st.title("FCM 模糊認知圖 - 策略模擬器")
st.markdown("---")

# ==========================================
# 核心運算公式 (你的研究方法核心)
# ==========================================
def sigmoid(x, lambd):
    """
    公式： A = 1 / (1 + e^(-lambda * x))
    將總輸入值轉化為 0~1 的狀態值
    """
    return 1 / (1 + np.exp(-lambd * x))

def run_fcm(W, A_init, lambd, steps, epsilon):
    history = [A_init]
    current_state = A_init
    
    for _ in range(steps):
        # 1. 矩陣運算 (狀態 x 權重)
        influence = np.dot(current_state, W)
        # 2. 公式轉換
        next_state = sigmoid(influence, lambd)
        
        history.append(next_state)
        
        # 判斷是否穩定 (收斂)
        if np.max(np.abs(next_state - current_state)) < epsilon:
            break
        current_state = next_state
        
    return np.array(history)

# ==========================================
# 介面設計：左側控制欄
# ==========================================
st.sidebar.header("系統設定")

# 1. 參數設定
LAMBDA = st.sidebar.slider("Lambda (敏感度)", 0.1, 5.0, 1.0, 0.1)
MAX_STEPS = st.sidebar.slider("最大模擬次數", 10, 100, 50, 5)
EPSILON = 0.001

st.sidebar.markdown("---")

# ==========================================
# PART 1: 矩陣 Excel (身體結構)
# ==========================================
st.header("第一部分：矩陣設定 (Matrix)")
st.info("請上傳包含權重矩陣的 Excel 檔案。若未上傳，系統將使用內建的 14x14 範例數據。")

uploaded_file = st.file_uploader("上傳 Excel 或 CSV 檔", type=['xlsx', 'csv'])

# 預設變數 (內建範例數據，讓你沒檔案也能跑)
if uploaded_file is None:
    # 這裡放的是你圖片辨識出來的 14 個概念
    concepts = [f"C{i+1}" for i in range(14)] 
    # 這是之前幫你辨識的矩陣 (為了版面整潔先隱藏細節，程式會讀取)
    # 這裡為了演示，先生成一個簡易的隨機矩陣，等你上傳 Excel 就會被蓋過去
    weights = np.zeros((14, 14)) 
    # 填入幾個關鍵數值示意
    weights[0, 1] = 0.65 # C1->C2
    weights[1, 2] = 0.8  # C2->C3
    st.warning("⚠️ 目前使用「內建測試矩陣」。若要進行正式研究，請上傳 Excel。")
else:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, index_col=0)
        else:
            df = pd.read_excel(uploaded_file, index_col=0)
        
        concepts = df.columns.tolist()
        weights = df.values
        st.success(f"✅ 成功讀取矩陣！共偵測到 {len(concepts)} 個概念。")
        with st.expander("點擊查看讀取到的矩陣數據"):
            st.dataframe(df)
            
    except Exception as e:
        st.error(f"檔案讀取錯誤: {e}")
        st.stop()

st.markdown("---")

# ==========================================
# PART 2: 初始值設定 (靈魂注入)
# ==========================================
st.header("第二部分：初始值設定 (Initial Values)")
st.markdown("請調整下方的拉桿，設定各概念的起始狀態 (0 = 無投入，1 = 全力投入)。這代表你的**策略情境**。")

# 建立 3 欄排列，讓拉桿不會拉太長
cols = st.columns(3)
initial_values = []

# 自動產生拉桿
for i, concept in enumerate(concepts):
    with cols[i % 3]: # 讓拉桿依序排列在 3 個欄位中
        val = st.slider(f"{concept}", 0.0, 1.0, 0.0, key=f"init_{i}")
        initial_values.append(val)

initial_state = np.array(initial_values)

# ==========================================
# 執行按鈕與結果
# ==========================================
st.markdown("---")
if st.button("🚀 開始運算 (Run Simulation)", type="primary"):
    
    # 呼叫上面的公式函數
    results = run_fcm(weights, initial_state, LAMBDA, MAX_STEPS, EPSILON)
    
    # --- 顯示結果 1: 趨勢圖 ---
    st.subheader("📊 模擬趨勢圖")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 只畫出「數值有變動」的概念，避免圖表太亂
    has_change = np.var(results, axis=0) > 0.0001
    active_concepts = [concepts[i] for i in range(len(concepts)) if has_change[i]]
    
    if len(active_concepts) == 0:
        st.warning("圖表無變化。原因可能是：所有初始值都設為 0，或者矩陣權重太小。")
    else:
        for i in range(len(concepts)):
            if has_change[i]: # 只畫有動的線
                ax.plot(results[:, i], label=concepts[i], marker='o', markersize=3, alpha=0.8)
        
        ax.set_xlabel("時間 (Steps)")
        ax.set_ylabel("激活程度 (Activation Level)")
        ax.set_title(f"FCM 動態模擬 (Lambda={LAMBDA})")
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

    # --- 顯示結果 2: 數據表 ---
    st.subheader("📋 最終穩定狀態數據")
    
    final_state = results[-1]
    # 計算「變動量」 (最終值 - 初始值)
    change = final_state - initial_state
    
    res_df = pd.DataFrame({
        "概念名稱": concepts,
        "初始投入": initial_state,
        "最終結果": final_state,
        "成長幅度": change
    }).sort_values(by="最終結果", ascending=False)
    
    # 用顏色標記數據 (深色代表數值高)
    st.dataframe(res_df.style.background_gradient(cmap='Blues', subset=['最終結果', '成長幅度']))

    # --- 下載功能 ---
    st.download_button(
        label="📥 下載本次模擬結果 (CSV)",
        data=pd.DataFrame(results, columns=concepts).to_csv().encode('utf-8'),
        file_name='simulation_result.csv',
        mime='text/csv'
    )
