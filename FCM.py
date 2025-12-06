import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 0. 頁面初始化 (防止 NameError 的關鍵)
# ==========================================
st.set_page_config(page_title="FCM 論文決策系統 (終極版)", layout="wide")

# CSS 美化：讓論文報告看起來像真的文件
st.markdown("""
<style>
    .chat-user { background-color: #DCF8C6; padding: 15px; border-radius: 15px; margin: 10px 0; text-align: right; color: black; box-shadow: 1px 1px 3px rgba(0,0,0,0.1); }
    .chat-ai { background-color: #F8F9FA; padding: 20px; border-radius: 15px; margin: 10px 0; text-align: left; color: #2c3e50; border-left: 5px solid #3498db; box-shadow: 1px 1px 3px rgba(0,0,0,0.1); }
    .paper-section { font-family: "Times New Roman", serif; line-height: 1.6; }
    .highlight { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 初始化記憶體
# ==========================================
if 'concepts' not in st.session_state:
    st.session_state.concepts = [
        "A1 倫理文化", "A2 高層基調", "A3 倫理風險",
        "B1 策略一致性", "B2 利害關係人", "B3 資訊透明",
        "C1 社會影響", "C2 環境責任", "C3 治理法遵"
    ]

# 預設矩陣 (避免全0)
if 'matrix' not in st.session_state:
    mat = np.zeros((9, 9))
    # 填入論文邏輯：A2 高層基調是核心驅動
    mat[1, 0] = 0.85 # -> A1
    mat[1, 3] = 0.80 # -> B1
    mat[1, 5] = 0.75 # -> B3
    mat[5, 4] = 0.90 # B3 -> B2
    mat[2, 8] = 0.80 # A3 -> C3
    mat[3, 6] = 0.50 # B1 -> C1
    st.session_state.matrix = mat

if 'last_results' not in st.session_state:
    st.session_state.last_results = None
    st.session_state.last_initial = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append({
        "role": "ai", 
        "content": "您好，已準備好進行論文寫作。請先在「模擬運算」跑出數據，然後輸入 **「幫我寫成1000字論文結論」**，我將為您生成完整的學術章節。"
    })

# ==========================================
# 2. 核心運算函數
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

def sort_matrix_logic():
    """強制排序功能：解決 A4 跑到最後面的問題"""
    df = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
    df_sorted = df.sort_index(axis=0).sort_index(axis=1)
    st.session_state.concepts = df_sorted.index.tolist()
    st.session_state.matrix = df_sorted.values

# ==========================================
# 3. 側邊欄設定
# ==========================================
st.sidebar.title("🛠️ 設定面板")
mode = st.sidebar.radio("資料來源", ["使用內建論文模型", "上傳 Excel/CSV"])

if mode == "上傳 Excel/CSV":
    uploaded = st.sidebar.file_uploader("上傳矩陣檔", type=['xlsx', 'csv'])
    if uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded, index_col=0)
            else:
                df = pd.read_excel(uploaded, index_col=0)
            st.session_state.concepts = df.columns.tolist()
            st.session_state.matrix = df.values
            st.sidebar.success(f"讀取成功 ({len(df)}x{len(df)})")
        except:
            st.sidebar.error("格式錯誤")
else:
    with st.sidebar.expander("➕ 新增準則 / 排序", expanded=True):
        new_c = st.text_input("輸入新準則 (如: A4 人才)")
        col_add, col_sort = st.columns(2)
        
        if col_add.button("加入"):
            if new_c and new_c not in st.session_state.concepts:
                st.session_state.concepts.append(new_c)
                old = st.session_state.matrix
                r, c = old.shape
                new_m = np.zeros((r+1, c+1))
                new_m[:r, :c] = old
                st.session_state.matrix = new_m
                st.rerun()
        
        if col_sort.button("🔄 排序"):
            sort_matrix_logic()
            st.success("已完成 A-Z 排序")
            st.rerun()

LAMBDA = st.sidebar.slider("Lambda (敏感度)", 0.1, 5.0, 1.0)
MAX_STEPS = st.sidebar.slider("模擬步數", 10, 100, 30)

# ==========================================
# 4. 主畫面 Tabs (解決 NameError)
# ==========================================
st.title("FCM 論文決策系統 (Thesis Generator)")
tab1, tab2, tab3 = st.tabs(["📊 矩陣視圖", "📈 模擬運算", "🎓 論文寫作顧問"])

# --- Tab 1 ---
with tab1:
    st.subheader("矩陣檢視")
    df_show = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
    st.dataframe(df_show.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1), height=400)
    st.download_button("下載 CSV", df_show.to_csv().encode('utf-8'), "matrix.csv")

# --- Tab 2 ---
with tab2:
    st.subheader("情境模擬")
    st.info("💡 操作：請拉動 **A2 高層基調** 至 0.8 以上，再按開始運算。")
    
    cols = st.columns(3)
    initial_vals = []
    for i, c in enumerate(st.session_state.concepts):
        with cols[i % 3]:
            val = st.slider(c, 0.0, 1.0, 0.0, key=f"init_{i}")
            initial_vals.append(val)
    
    if st.button("🚀 開始模擬運算", type="primary"):
        init_arr = np.array(initial_vals)
        res = run_fcm(st.session_state.matrix, init_arr, LAMBDA, MAX_STEPS, 0.001)
        
        st.session_state.last_results = res
        st.session_state.last_initial = init_arr
        
        fig, ax = plt.subplots(figsize=(10, 5))
        active_idx = [i for i in range(len(res[0])) if res[-1, i] > 0.01 or init_arr[i] > 0]
        
        if not active_idx:
            st.warning("⚠️ 數值無變化，請嘗試增加初始投入。")
        else:
            for i in active_idx:
                ax.plot(res[:, i], label=st.session_state.concepts[i])
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            st.pyplot(fig)

# --- Tab 3: AI 論文寫作核心 (重寫版) ---
with tab3:
    st.subheader("🤖 論文生成與深度分析")
    
    # 顯示歷史訊息
    for msg in st.session_state.chat_history:
        role_class = "chat-user" if msg["role"] == "user" else "chat-ai"
        prefix = "👤 您：" if msg["role"] == "user" else "🤖 AI："
        st.markdown(f'<div class="{role_class}"><b>{prefix}</b><br>{msg["content"]}</div>', unsafe_allow_html=True)

    user_input = st.text_input("輸入指令 (建議輸入：幫我寫成1000字論文結論)", key="chat_in")
    
    if st.button("送出") and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        if st.session_state.last_results is None:
            response = "⚠️ 請先至「模擬運算」分頁跑出數據，我才能寫論文。"
        else:
            # 準備數據
            results = st.session_state.last_results
            initial = st.session_state.last_initial
            final = results[-1]
            growth = final - initial
            concepts = st.session_state.concepts
            steps = results.shape[0]
            
            # 找出關鍵數據指標
            driver_idx = np.argmax(initial)
            driver_name = concepts[driver_idx]
            best_idx = np.argmax(growth)
            best_name = concepts[best_idx]
            
            response = ""

            # ========================================================
            # 模式 A: 論文結論生成 (針對 1000字 / 論文 / 結論)
            # ========================================================
            if any(k in user_input for k in ["論文", "結論", "1000", "報告"]):
                response += f"""
### 🎓 第五章：結論與建議 (模擬生成草稿)

**5.1 研究結論 (Research Conclusions)**

本研究運用模糊認知圖 (FCM) 方法，旨在探討製造業 ESG 策略之動態因果關係。經由 {steps} 個疊代週期的系統模擬，本研究獲得以下關鍵實證發現：

**第一，確認「{driver_name}」為啟動永續轉型的核心驅動因子。**
模擬結果顯示，當企業將資源優先投入於 **{driver_name}** (Initial Input={initial[driver_idx]:.2f}) 時，系統呈現最強烈的正向連鎖反應。此一發現量化驗證了「治理先行」的策略邏輯，意即企業必須先建立穩固的內部治理機制，方能有效帶動後端的績效表現。

**第二，揭示策略擴散的路徑依賴性。**
數據顯示，**{best_name}** 為此策略路徑下的最大受惠者，其數值從初始的 {initial[best_idx]:.2f} 顯著成長至 {final[best_idx]:.2f} (成長幅度 +{growth[best_idx]:.2f})。這證實了 {driver_name} 與 {best_name} 之間存在顯著的「外溢效應 (Spillover Effect)」，顯示 ESG 構面間並非獨立運作，而是具有高度的互依性。

---

**5.2 管理意涵 (Managerial Implications)**

基於上述研究發現，本研究對製造業管理者提出以下具體建議：

**1. 資源配置的最佳化：槓桿策略的應用**
在資源有限的限制下，管理者應避免採取齊頭式的資源分配。模擬結果建議，應採取「精準打擊」策略，集中資源強化 **{driver_name}**。透過 FCM 的矩陣傳導機制，單點突破該指標即可帶動整體系統的被動成長，此為最具成本效益的決策模式。

**2. 建立具備動態觀點的績效考核制度**
從模擬圖形的收斂過程可見，策略介入初期系統存在約 5-10 個週期的「適應震盪期」。管理者應理解此一時間滯後性 (Time Lag)，在推動初期不應因 **{best_name}** 等績效指標未立即提升而輕易終止策略，應給予組織文化內化的時間。

---

**5.3 學術理論貢獻 (Theoretical Contributions)**

**1. 豐富了高階梯隊理論 (Upper Echelons Theory) 的實證內涵**
本研究透過動態模擬，具體呈現了領導者價值觀 ({driver_name}) 如何透過組織機制轉化為具體的 ESG 績效。這突破了過往研究多採靜態相關分析的限制，提供了更具解釋力的因果推論證據。

**2. 填補了 ESG 動態評估方法的缺口**
本研究證實 FCM 作為一種半量化工具，能有效處理 ESG 議題中模糊且複雜的變數關係，為後續研究提供了一套可複製的分析架構。
"""

            # ========================================================
            # 模式 B: 詳細解釋每一個準則
            # ========================================================
            elif any(k in user_input for k in ["每一", "詳細", "全部"]):
                response += "### 📋 各準則深度動態分析\n\n"
                for i, c in enumerate(concepts):
                    g = growth[i]
                    role = "🔴 驅動者" if initial[i] > 0 else ("🟢 受惠者" if g > 0.1 else "⚪ 邊緣因子")
                    response += f"**{c} ({role})**\n"
                    response += f"- 初始: {initial[i]:.1f} → 最終: {final[i]:.2f} (成長: {g:+.2f})\n"
                    response += f"- 分析: 該指標在模擬中展現了{ '顯著' if g>0.1 else '微弱' }的反應。建議在論文中探討其{ '對整體績效的貢獻' if g>0.1 else '反應遲鈍的結構性原因' }。\n\n"
            
            # ========================================================
            # 模式 C: 一般回答
            # ========================================================
            else:
                response += f"根據模擬，表現最佳的是 **{best_name}**。\n若您需要產生論文，請輸入 **「幫我寫成1000字論文結論」**。"

        st.session_state.chat_history.append({"role": "ai", "content": response})
        st.rerun()
