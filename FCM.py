import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 系統設定
# ==========================================
st.set_page_config(page_title="ESG 策略動態模擬", layout="wide")
st.title("台灣製造業 ESG 策略模擬器 (基於論文邏輯)")
st.markdown("""
### 矩陣邏輯說明 (依據賴育津論文推論)
* **核心驅動力**：設定 **A2 高層基調** 對所有治理與策略構面有強烈正向影響 (0.8~0.9)。
* **傳導路徑**：治理 (A) $\\rightarrow$ 策略 (B) $\\rightarrow$ 績效 (C)。
* **負向反饋**：模擬真實世界資源排擠，若過度僅關注短期策略一致性，可能對某些創新投入有微弱負影響 (範例設定)。
""")

# 定義 9 大準則
concepts = [
    "A1 倫理文化", "A2 高層基調", "A3 倫理風險",
    "B1 策略一致性", "B2 利害關係人", "B3 資訊透明",
    "C1 社會影響", "C2 環境責任", "C3 治理法遵"
]

# ==========================================
# 2. 內建矩陣數據 (根據論文邏輯填滿)
# ==========================================
# Row (因) -> Column (果)
# 例如 weights[1, 0] = 0.85 代表 A2(高層) 強烈影響 A1(文化)
weights = np.array([
    # A1,   A2,   A3,   B1,   B2,   B3,   C1,   C2,   C3
    [0.0,  0.3,  0.6,  0.5,  0.4,  0.0,  0.2,  0.0,  0.7], # A1 倫理文化
    [0.85, 0.0,  0.9,  0.8,  0.5,  0.7,  0.0,  0.0,  0.6], # A2 高層基調 (最強驅動因子)
    [0.5,  0.2,  0.0,  0.4,  0.0,  0.6,  0.0,  0.3,  0.8], # A3 倫理風險 (直接影響法遵 C3)
    [0.0,  0.0,  0.0,  0.0,  0.6,  0.5,  0.4,  0.5,  0.0], # B1 策略一致性
    [0.3,  0.0,  0.0,  0.5,  0.0,  0.8,  0.7,  0.0,  0.0], # B2 利害關係人 (影響社會 C1)
    [0.2,  0.0,  0.4,  0.0,  0.9,  0.0,  0.5,  0.0,  0.0], # B3 資訊透明 (強烈影響利害關係人 B2)
    [0.0,  0.2,  0.0,  0.0,  0.5,  0.3,  0.0,  0.1,  0.0], # C1 社會影響 (反饋)
    [-0.1, 0.0, -0.2,  0.0,  0.0,  0.0,  0.2,  0.0,  0.0], # C2 環境責任 (設一點負值代表成本壓力)
    [0.4,  0.5,  0.0,  0.0,  0.0,  0.4,  0.0,  0.0,  0.0]  # C3 治理法遵 (績效好會回頭強化信任)
])

# ==========================================
# 3. 核心公式
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
# 4. 側邊欄設定
# ==========================================
st.sidebar.header("參數控制")
LAMBDA = st.sidebar.slider("Lambda (敏感度)", 0.1, 5.0, 1.0, 0.1)
MAX_STEPS = st.sidebar.slider("模擬步數", 10, 100, 40, 5)
EPSILON = 0.001

st.sidebar.markdown("---")
st.sidebar.header("情境設定 (初始投入)")
st.sidebar.info("試著把 **A2 高層基調** 拉到 1.0，觀察它如何帶動其他線條上升。")

cols = st.columns(3)
initial_values = []
for i, concept in enumerate(concepts):
    with cols[i % 3]:
        # 預設值全為 0，讓使用者自己拉，這樣比較有感
        val = st.slider(f"{concept}", 0.0, 1.0, 0.0, key=f"init_{i}")
        initial_values.append(val)
initial_state = np.array(initial_values)

# ==========================================
# 5. 執行與繪圖
# ==========================================
if st.button("🚀 開始模擬 (Run)", type="primary"):
    
    results = run_fcm(weights, initial_state, LAMBDA, MAX_STEPS, EPSILON)
    
    # 畫圖
    st.subheader("動態趨勢圖")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 檢查有沒有任何數值變化
    if np.all(results[-1] == results[0]):
        st.warning("⚠️ 警告：所有初始值都是 0，系統沒有動力。請在上方拉桿至少設定一個概念為 1.0 (例如 A2)。")
    else:
        # 使用不同顏色與線型讓圖表更豐富
        styles = ['-', '--', '-.', ':']
        for i, concept in enumerate(concepts):
            # 只畫出最終有被激活的概念
            if results[-1, i] > 0.1:
                ax.plot(results[:, i], label=concept, linestyle=styles[i % 4], linewidth=2)
        
        ax.set_title(f"ESG FCM Simulation (Lambda={LAMBDA})")
        ax.set_ylabel("Activation Level (0-1)")
        ax.set_xlabel("Time Steps")
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # 數據表
        st.subheader("最終收斂數據")
        final_state = results[-1]
        df_res = pd.DataFrame({
            "準則": concepts,
            "初始值": initial_state,
            "最終值": final_state,
            "成長幅度": final_state - initial_state
        }).sort_values(by="最終值", ascending=False)
        st.dataframe(df_res.style.background_gradient(cmap='Greens'))
        
    # 提供這個內建矩陣的下載
    st.markdown("---")
    st.write("覺得這個內建矩陣不錯？你可以下載回去 Excel 修改：")
    df_export = pd.DataFrame(weights, index=concepts, columns=concepts)
    st.download_button(
        "📥 下載此預設矩陣 (CSV)",
        df_export.to_csv().encode('utf-8'),
        "esg_fcm_matrix_full.csv",
        "text/csv"
    )
