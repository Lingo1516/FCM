import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 0. 初始化狀態 (Session State)
# 這是為了讓你的「新增準則」不會因為按了別的按鈕就消失
# ==========================================
if 'concepts' not in st.session_state:
    # [cite_start]預設來自論文的 9 大準則 [cite: 88, 93-102]
    st.session_state.concepts = [
        "A1 倫理文化", "A2 高層基調", "A3 倫理風險",
        "B1 策略一致性", "B2 利害關係人", "B3 資訊透明",
        "C1 社會影響", "C2 環境責任", "C3 治理法遵"
    ]

if 'matrix' not in st.session_state:
    # 初始化 9x9 矩陣
    st.session_state.matrix = np.zeros((9, 9))
    # 填入論文邏輯的預設值 (作為起始點)
    weights = st.session_state.matrix
    weights[1, 0] = 0.85 # A2->A1
    weights[1, 3] = 0.8  # A2->B1
    weights[1, 5] = 0.7  # A2->B3
    weights[2, 8] = 0.8  # A3->C3
    weights[5, 4] = 0.9  # B3->B2

# ==========================================
# 1. 頁面設定
# ==========================================
st.set_page_config(page_title="FCM 高階模擬器", layout="wide")
st.title("FCM 動態策略模擬器 (可編輯版)")

# ==========================================
# 2. 側邊欄：資料來源控制
# ==========================================
st.sidebar.header("1. 資料來源設定")

data_source = st.sidebar.radio(
    "請選擇矩陣來源：",
    ("📂 上傳 Excel 檔案", "🎲 使用內建/隨機模擬")
)

# --- 模式 A: 上傳檔案 ---
if data_source == "📂 上傳 Excel 檔案":
    uploaded_file = st.sidebar.file_uploader("上傳 Excel (.xlsx) 或 CSV", type=['xlsx', 'csv'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, index_col=0)
            else:
                df = pd.read_excel(uploaded_file, index_col=0)
            
            # 更新系統狀態
            st.session_state.concepts = df.columns.tolist()
            st.session_state.matrix = df.values
            st.sidebar.success(f"讀取成功！矩陣大小: {df.shape}")
        except Exception as e:
            st.sidebar.error(f"檔案格式錯誤: {e}")

# --- 模式 B: 內建/隨機模擬 ---
else:
    st.sidebar.subheader("模擬矩陣控制")
    
    col_rand1, col_rand2 = st.sidebar.columns(2)
    
    # 功能：隨機生成矩陣
    if col_rand1.button("🎲 隨機生成權重"):
        n = len(st.session_state.concepts)
        # 生成 -0.5 到 0.8 之間的隨機數
        rand_matrix = np.random.uniform(-0.5, 0.8, (n, n))
        # 對角線設為 0 (自己不影響自己，通常 FCM 的設定)
        np.fill_diagonal(rand_matrix, 0)
        # 過濾太小的雜訊 (讓矩陣稀疏一點，比較像真實世界)
        rand_matrix[np.abs(rand_matrix) < 0.2] = 0
        
        st.session_state.matrix = rand_matrix
        st.sidebar.success("已生成隨機矩陣！")

    # 功能：重置回論文預設值
    if col_rand2.button("↺ 重置為預設"):
        n = 9
        st.session_state.concepts = [
            "A1 倫理文化", "A2 高層基調", "A3 倫理風險",
            "B1 策略一致性", "B2 利害關係人", "B3 資訊透明",
            "C1 社會影響", "C2 環境責任", "C3 治理法遵"
        ]
        new_mat = np.zeros((9, 9))
        new_mat[1, [0,3,5]] = [0.85, 0.8, 0.7] # A2 的影響
        st.session_state.matrix = new_mat
        st.rerun()

    # --- 功能：動態新增準則 ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("➕ 新增自訂準則")
    new_concept_name = st.sidebar.text_input("輸入新準則名稱 (例如: D1 創新)", "")
    
    if st.sidebar.button("加入矩陣"):
        if new_concept_name and new_concept_name not in st.session_state.concepts:
            # 1. 增加名稱
            st.session_state.concepts.append(new_concept_name)
            
            # 2. 擴充矩陣 (舊的保留，新增的一行一列補 0)
            old_matrix = st.session_state.matrix
            rows, cols = old_matrix.shape
            # 建立大一號的 0 矩陣
            new_matrix = np.zeros((rows + 1, cols + 1))
            # 把舊數據貼回去左上角
            new_matrix[:rows, :cols] = old_matrix
            
            # 更新狀態
            st.session_state.matrix = new_matrix
            st.sidebar.success(f"已新增: {new_concept_name}")
            st.rerun() # 重新整理頁面以顯示新拉桿
        elif new_concept_name in st.session_state.concepts:
            st.sidebar.warning("該準則名稱已存在！")

# ==========================================
# 3. 矩陣預覽與編輯提示
# ==========================================
with st.expander("點擊查看目前矩陣數值 (Matrix View)", expanded=False):
    df_display = pd.DataFrame(
        st.session_state.matrix, 
        columns=st.session_state.concepts, 
        index=st.session_state.concepts
    )
    st.dataframe(df_display.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1))
    st.caption("提示：如果是隨機生成的矩陣，紅色代表負相關，藍色代表正相關。")

# ==========================================
# 4. 參數與情境設定 (中間主區塊)
# ==========================================
st.markdown("---")
col_param, col_init = st.columns([1, 2])

with col_param:
    st.subheader("參數設定")
    LAMBDA = st.slider("Lambda (敏感度)", 0.1, 5.0, 1.0)
    MAX_STEPS = st.slider("模擬步數", 10, 100, 30)
    EPSILON = 0.001

with col_init:
    st.subheader("情境設定 (初始投入)")
    st.info("請調整下方拉桿，設定各準則的起始狀態 (0~1)。")
    
    # 動態生成拉桿 (根據目前的 concepts 數量)
    initial_values = []
    # 使用 columns 排版，每行 3 個
    cols = st.columns(3)
    for i, concept in enumerate(st.session_state.concepts):
        with cols[i % 3]:
            val = st.slider(f"{concept}", 0.0, 1.0, 0.0, key=f"init_{i}")
            initial_values.append(val)
    
    initial_state = np.array(initial_values)

# ==========================================
# 5. 核心運算公式
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
# 6. 執行與結果
# ==========================================
st.markdown("---")
if st.button("🚀 開始模擬 (Run Simulation)", type="primary"):
    
    # 使用 session_state 中的矩陣進行運算
    W = st.session_state.matrix
    results = run_fcm(W, initial_state, LAMBDA, MAX_STEPS, EPSILON)
    
    # 繪圖
    st.subheader("動態趨勢圖")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 檢查有沒有數值變動
    has_change = np.var(results, axis=0) > 0.00001
    active_indices = [i for i, x in enumerate(has_change) if x]
    
    if len(active_indices) == 0:
        st.warning("⚠️ 圖表無變化。可能原因：(1) 初始值全為 0，(2) 矩陣權重全為 0。建議嘗試「隨機生成權重」或拉動初始值。")
    else:
        for i in active_indices:
            concept_name = st.session_state.concepts[i]
            ax.plot(results[:, i], label=concept_name, marker='o', markersize=3, alpha=0.8)
            
        ax.set_title(f"FCM Simulation (Concepts: {len(st.session_state.concepts)})")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Activation Level")
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # 數據下載區
    st.subheader("數據導出")
    res_df = pd.DataFrame(results, columns=st.session_state.concepts)
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            "📥 下載模擬結果 (Result CSV)",
            res_df.to_csv().encode('utf-8'),
            "simulation_result.csv",
            "text/csv"
        )
    with col_d2:
        # 讓使用者下載目前的矩陣 (包含隨機生成或新增準則後的矩陣)
        current_matrix_df = pd.DataFrame(
            st.session_state.matrix,
            index=st.session_state.concepts,
            columns=st.session_state.concepts
        )
        st.download_button(
            "📥 下載目前矩陣 (Matrix CSV)",
            current_matrix_df.to_csv().encode('utf-8'),
            "current_matrix.csv",
            "text/csv"
        )
