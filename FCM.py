import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 系統設定與概念定義 (來自你的論文)
# ==========================================
st.set_page_config(page_title="ESG 倫理治理 FCM 分析", layout="wide")
st.title("台灣製造業 ESG 策略核心 - FCM 分析模型")
st.markdown("""
### 模型架構說明
本模型採用論文定義之 **3大構面** 與 **9項準則** 作為系統節點：
* **構面 A (倫理治理)**：A1 倫理文化, A2 高層基調, A3 倫理風險管理
* **構面 B (ESG策略整合)**：B1 策略一致性, B2 利害關係人參與, B3 資訊透明揭露
* **構面 C (責任績效)**：C1 社會影響力, C2 環境責任, C3 治理與法遵績效
""")

# 論文中的 9 個準則名稱
concepts = [
    "A1 倫理文化", 
    "A2 高層基調", 
    "A3 倫理風險管理",
    "B1 策略一致性", 
    "B2 利害關係人參與", 
    "B3 資訊透明揭露",
    "C1 社會影響力", 
    "C2 環境責任", 
    "C3 治理與法遵績效"
]

# ==========================================
# 2. 核心運算公式
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
# 3. 側邊欄：參數與矩陣設定
# ==========================================
st.sidebar.header("1. 參數設定")
LAMBDA = st.sidebar.slider("Lambda (敏感度)", 0.1, 5.0, 1.0, 0.1)
MAX_STEPS = st.sidebar.slider("最大模擬次數", 10, 100, 50, 5)
EPSILON = 0.001

st.sidebar.markdown("---")
st.sidebar.header("2. 矩陣來源 (Matrix)")

# 選項：使用範例或上傳
matrix_source = st.sidebar.radio("選擇矩陣來源：", ["使用預設範例 (9x9)", "上傳 Excel/CSV"])

if matrix_source == "上傳 Excel/CSV":
    uploaded_file = st.sidebar.file_uploader("上傳檔案", type=['xlsx', 'csv'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, index_col=0)
            else:
                df = pd.read_excel(uploaded_file, index_col=0)
            weights = df.values
            concepts = df.columns.tolist() # 如果檔案裡有名稱，以檔案為主
            st.sidebar.success(f"讀取成功！({len(concepts)}x{len(concepts)})")
        except:
            st.sidebar.error("格式錯誤，請確保第一列與第一欄為概念名稱")
            weights = np.zeros((9, 9)) # 防呆
    else:
        weights = np.zeros((9, 9)) # 尚未上傳時的空矩陣
        
else:
    # --- 建立一個 9x9 的範例矩陣 ---
    # 這裡我先填入一些假設數值，你需要根據研究填入真實的影響權重
    weights = np.zeros((9, 9))
    
    # 範例邏輯：假設「A2 高層基調」會強烈影響「A1 倫理文化」和「B1 策略一致性」
    # (這符合論文觀點：高層基調是關鍵驅動因子 [cite: 126])
    weights[1, 0] = 0.8  # A2 -> A1 (強影響)
    weights[1, 3] = 0.7  # A2 -> B1
    weights[1, 5] = 0.6  # A2 -> B3 (透明揭露)
    
    # 假設「B3 透明揭露」會影響「C1 社會影響力」
    weights[5, 6] = 0.5 
    
    st.sidebar.info("目前使用內建 9x9 測試矩陣 (基於論文邏輯的假設值)。")

# ==========================================
# 4. 初始值設定 (情境模擬)
# ==========================================
st.header("情境模擬設定 (Initial States)")
st.info("請調整下方拉桿，模擬當某個策略被啟動時 (例如高層基調 = 1)，對整體績效的影響。")

cols = st.columns(3)
initial_values = []

for i, concept in enumerate(concepts):
    with cols[i % 3]:
        # 預設把 A2 (高層基調) 設高一點，因為論文說它是最重要的 [cite: 127]
        default_val = 0.5
        if "A2" in concept: 
            default_val = 0.0
            
        val = st.slider(f"{concept}", 0.0, 1.0, default_val, key=f"init_{i}")
        initial_values.append(val)

initial_state = np.array(initial_values)

# ==========================================
# 5. 執行與結果顯示
# ==========================================
st.markdown("---")
if st.button("🚀 開始分析 (Run Analysis)", type="primary"):
    
    results = run_fcm(weights, initial_state, LAMBDA, MAX_STEPS, EPSILON)
    
    # 1. 趨勢圖
    st.subheader("📊 動態趨勢圖")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 設定中文字型 (為了讓 Streamlit Cloud 盡量顯示，使用通用設定)
    # 如果是本地端跑，可以解開下面這行
    # plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    
    for i, concept in enumerate(concepts):
        # 為了簡化圖表，只畫出最終數值 > 0.01 的線
        if results[-1, i] > 0.01 or initial_state[i] > 0:
            ax.plot(results[:, i], label=concept, marker='o', markersize=3)
            
    ax.set_xlabel("Steps")
    ax.set_ylabel("Activation")
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)
    
    # 2. 數據表
    st.subheader("📋 模擬結果數據")
    final_state = results[-1]
    res_df = pd.DataFrame({
        "準則名稱": concepts,
        "初始投入": initial_state,
        "最終產出": final_state,
        "變化量": final_state - initial_state
    }).sort_values(by="最終產出", ascending=False)
    
    st.dataframe(res_df.style.background_gradient(cmap='Greens'))

    # 3. 下載範例矩陣功能 (方便你建立 Excel)
    st.markdown("---")
    st.subheader("🛠️ 工具區")
    st.write("還沒有矩陣檔嗎？下載這個範例，填入你的專家權重後再上傳：")
    
    # 建立 9x9 空白範例
    example_df = pd.DataFrame(np.zeros((9, 9)), index=concepts, columns=concepts)
    csv = example_df.to_csv().encode('utf-8')
    
    st.download_button(
        "📥 下載 9x9 矩陣範本 (CSV)",
        csv,
        "ESG_FCM_Template.csv",
        "text/csv"
    )
