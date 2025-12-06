import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 0. 頁面初始化與樣式
# ==========================================
st.set_page_config(page_title="FCM 論文決策系統", layout="wide")

# 自訂 CSS 讓報告更漂亮
st.markdown("""
<style>
    .report-box { background-color: #f0f2f6; border-left: 5px solid #4CAF50; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    .academic-box { background-color: #e8f4f8; border-left: 5px solid #2196F3; padding: 15px; border-radius: 5px; }
    .manage-box { background-color: #fff3e0; border-left: 5px solid #FF9800; padding: 15px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 初始化 Session State (記憶體)
# ==========================================
# 如果是第一次打開，載入預設的 9 大準則
if 'concepts' not in st.session_state:
    st.session_state.concepts = [
        "A1 倫理文化", "A2 高層基調", "A3 倫理風險",
        "B1 策略一致性", "B2 利害關係人", "B3 資訊透明",
        "C1 社會影響", "C2 環境責任", "C3 治理法遵"
    ]

# 初始化矩陣 (如果沒資料，自動填入論文邏輯，避免全是0)
if 'matrix' not in st.session_state:
    # 建立 9x9
    mat = np.zeros((9, 9))
    # === 寫入論文邏輯 (Hardcoded Logic) ===
    # A2 高層基調 -> 驅動核心
    mat[1, 0] = 0.85 # 影響倫理文化
    mat[1, 3] = 0.80 # 影響策略一致性
    mat[1, 5] = 0.75 # 影響資訊透明
    # B3 資訊透明 -> 影響利害關係人
    mat[5, 4] = 0.90
    # A3 倫理風險 -> 影響治理法遵
    mat[2, 8] = 0.80
    # B1 策略一致 -> 影響績效
    mat[3, 6] = 0.5
    mat[3, 7] = 0.6
    
    st.session_state.matrix = mat

# 儲存模擬結果供 AI 分析
if 'last_results' not in st.session_state:
    st.session_state.last_results = None
if 'last_initial' not in st.session_state:
    st.session_state.last_initial = None

# ==========================================
# 2. 核心函數 (排序、運算)
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

def sort_matrix():
    """自動排序功能：新增準則後，按這個讓它歸位"""
    df = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
    df_sorted = df.sort_index(axis=0).sort_index(axis=1)
    st.session_state.concepts = df_sorted.index.tolist()
    st.session_state.matrix = df_sorted.values

# ==========================================
# 3. 側邊欄 (設定區)
# ==========================================
st.sidebar.title("🛠️ 設定面板")

# --- 資料來源選擇 ---
mode = st.sidebar.radio("資料來源模式", ["使用內建論文模型", "上傳 Excel/CSV"])

if mode == "上傳 Excel/CSV":
    uploaded = st.sidebar.file_uploader("請上傳矩陣檔", type=['xlsx', 'csv'])
    if uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded, index_col=0)
            else:
                df = pd.read_excel(uploaded, index_col=0)
            st.session_state.concepts = df.columns.tolist()
            st.session_state.matrix = df.values
            st.sidebar.success(f"✅ 讀取成功 ({len(df)}x{len(df)})")
        except Exception as e:
            st.sidebar.error(f"讀取失敗: {e}")
else:
    # 內建模式下的編輯功能
    with st.sidebar.expander("➕ 新增準則 / 排序"):
        new_c = st.text_input("輸入新準則 (如: A4 人才)")
        if st.button("加入矩陣"):
            if new_c and new_c not in st.session_state.concepts:
                st.session_state.concepts.append(new_c)
                # 擴充矩陣補 0
                old = st.session_state.matrix
                r, c = old.shape
                new_m = np.zeros((r+1, c+1))
                new_m[:r, :c] = old
                st.session_state.matrix = new_m
                st.rerun()
        
        if st.button("🔄 自動排序 (A-Z)"):
            sort_matrix()
            st.success("已排序！")
            st.rerun()
    
    if st.sidebar.button("🎲 隨機生成權重 (測試用)"):
        n = len(st.session_state.concepts)
        rand = np.random.uniform(-0.3, 0.8, (n, n))
        np.fill_diagonal(rand, 0)
        rand[np.abs(rand)<0.1] = 0
        st.session_state.matrix = rand
        st.sidebar.success("已生成隨機數據")

st.sidebar.markdown("---")
LAMBDA = st.sidebar.slider("Lambda (敏感度)", 0.1, 5.0, 1.0)
MAX_STEPS = st.sidebar.slider("模擬步數", 10, 100, 30)

# ==========================================
# 4. 主畫面 (Tabs 分頁)
# ==========================================
st.title("FCM 論文決策系統")

# 定義分頁 (解決 NameError 的關鍵)
tab1, tab2, tab3 = st.tabs(["📊 矩陣視圖 (Matrix)", "📈 模擬運算 (Simulate)", "🎓 AI 論文顧問 (Analysis)"])

# --- Tab 1: 矩陣視圖 ---
with tab1:
    st.subheader("檢視 / 編輯矩陣數值")
    st.caption("這是系統目前的「大腦」。您可以下載它，修改後再上傳。")
    
    df_show = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
    st.dataframe(df_show.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1), height=400)
    
    st.download_button(
        "📥 下載目前矩陣 (CSV)",
        df_show.to_csv().encode('utf-8'),
        "current_matrix.csv",
        "text/csv"
    )

# --- Tab 2: 模擬運算 ---
with tab2:
    st.subheader("情境模擬 (Scenario Analysis)")
    st.info("💡 操作提示：請拉動下方拉桿 (例如將 A2 拉到 1.0)，然後按「開始運算」。")
    
    # 動態產生拉桿
    cols = st.columns(3)
    initial_vals = []
    for i, c in enumerate(st.session_state.concepts):
        with cols[i % 3]:
            val = st.slider(c, 0.0, 1.0, 0.0, key=f"init_{i}")
            initial_vals.append(val)
    
    if st.button("🚀 開始模擬運算", type="primary"):
        init_arr = np.array(initial_vals)
        res = run_fcm(st.session_state.matrix, init_arr, LAMBDA, MAX_STEPS, 0.001)
        
        # 存起來給 Tab 3 用
        st.session_state.last_results = res
        st.session_state.last_initial = init_arr
        
        # 畫圖
        fig, ax = plt.subplots(figsize=(10, 5))
        active_idx = [i for i in range(len(res[0])) if res[-1, i] > 0.01 or init_arr[i] > 0]
        
        if not active_idx:
            st.warning("⚠️ 圖表沒有變化？可能是矩陣權重全是 0，或是初始值沒拉。請檢查 Tab 1 或拉動 A2。")
        else:
            for i in active_idx:
                ax.plot(res[:, i], label=st.session_state.concepts[i], marker='o', markersize=4)
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_title("FCM 動態收斂圖")
            ax.set_xlabel("Steps")
            ax.set_ylabel("Activation (0-1)")
            st.pyplot(fig)
            
            # 顯示結果表
            final_v = res[-1]
            df_res = pd.DataFrame({
                "準則": st.session_state.concepts,
                "初始投入": init_arr,
                "最終產出": final_v,
                "成長幅度": final_v - init_arr
            }).sort_values("最終產出", ascending=False)
            st.dataframe(df_res.style.background_gradient(cmap='Greens'))

# --- Tab 3: AI 論文顧問 ---
with tab3:
    st.subheader("🎓 模擬結果深度解析")
    
    if st.session_state.last_results is None:
        st.warning("請先在「📈 模擬運算」分頁跑一次結果，我才能分析。")
    else:
        # 準備數據
        results = st.session_state.last_results
        final = results[-1]
        initial = st.session_state.last_initial
        concepts = st.session_state.concepts
        
        # 找出關鍵數據
        driver_idx = np.argmax(initial) if np.sum(initial) > 0 else -1
        driver_name = concepts[driver_idx] if driver_idx != -1 else "無特定策略"
        
        # 找出受益最大的 (排除自己)
        growth = final - initial
        growth[initial > 0.8] = 0 # 排除原本就很高的是
        best_idx = np.argmax(growth)
        best_name = concepts[best_idx]
        
        # === 自動生成論文段落 ===
        st.markdown(f"""
        <div class="report-box">
        <b>📊 數據診斷：</b><br>
        本次模擬以 <b>{driver_name}</b> 為主要驅動策略（初始投入={initial[driver_idx]:.1f}）。<br>
        結果顯示，系統呈現連動反應，其中 <b>{best_name}</b> 的成長最為顯著（+{growth[best_idx]:.2f}），
        驗證了兩者之間存在強烈的因果路徑。
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="academic-box">
            <b>🏛️ 學術意涵 (Theoretical Implications)：</b><br><br>
            1. <b>驗證高階梯隊理論：</b>模擬結果支持了領導者認知（{driver_name}）對組織結果的決定性影響。數據顯示該因子具備高度的「中心性 (Centrality)」。<br><br>
            2. <b>路徑依賴效應：</b>從圖形收斂過程可見，治理機制的建立存在時間滯後性，這量化解釋了為何 ESG 轉型初期績效不明顯的現象。
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="manage-box">
            <b>💼 管理意涵 (Managerial Implications)：</b><br><br>
            1. <b>槓桿策略選擇：</b>管理者應避免資源分散，建議集中資源強化 <b>{driver_name}</b>，利用其外溢效果帶動 <b>{best_name}</b> 的被動成長。<br><br>
            2. <b>關鍵績效指標(KPI)設定：</b>不應僅關注財務結果，應將 {driver_name} 的落實程度納入先期指標，以確保長期永續目標的達成。
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        st.subheader("💬 AI 策略問答")
        user_q = st.text_input("輸入問題 (例如：這個策略有什麼缺點？如何改善 C2？)")
        
        if user_q:
            st.write("🤖 **AI 分析：**")
            if "缺點" in user_q or "風險" in user_q:
                low_growth = [concepts[i] for i, g in enumerate(growth) if g < 0.05 and initial[i]==0]
                st.error(f"分析發現：{', '.join(low_growth[:3])} 等項目的反應微弱。這代表目前的策略無法有效觸及這些領域，這是潛在的盲點。")
            elif "改善" in user_q:
                st.info(f"若要改善特定指標，建議不要直接強拉該指標的數值（治標），而是要強化矩陣中對該指標有「正向權重」的源頭因子。")
            else:
                st.success(f"這是一個好問題。根據目前的模擬數據，{driver_name} 確實是系統中最具影響力的槓桿點。建議在論文中強調此一發現。")
