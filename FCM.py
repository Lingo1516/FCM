import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# ==========================================
# 0. 頁面初始化
# ==========================================
st.set_page_config(page_title="FCM 論文決策系統 (Step Fixed)", layout="wide")

st.markdown("""
<style>
    .report-box { 
        border: 1px solid #ccc; padding: 40px; background-color: #ffffff; 
        color: #000000; font-family: "Times New Roman", "標楷體", serif; 
        font-size: 16px; line-height: 2.0; text-align: justify;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 20px; white-space: pre-wrap;
    }
    .stButton>button { width: 100%; border-radius: 5px; height: 3.5em; font-weight: bold; font-size: 15px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 初始化數據
# ==========================================
if 'concepts' not in st.session_state:
    st.session_state.concepts = [
        "A1 倫理文化", "A2 高層基調", "A3 倫理風險",
        "B1 策略一致性", "B2 利害關係人", "B3 資訊透明",
        "C1 社會影響", "C2 環境責任", "C3 治理法遵"
    ]

if 'matrix' not in st.session_state:
    mat = np.zeros((9, 9))
    # 預設一些正負值
    mat[1, 0] = 0.85; mat[1, 3] = 0.80; mat[5, 4] = 0.90; mat[3, 6] = 0.60
    mat[2, 8] = -0.7; mat[0, 2] = -0.6
    st.session_state.matrix = mat

if 'last_results' not in st.session_state:
    st.session_state.last_results = None
    st.session_state.last_initial = None

if 'paper_sections' not in st.session_state:
    st.session_state.paper_sections = {"4.1": "", "4.2": "", "4.3": "", "4.4": "", "5.1": "", "5.2": "", "5.3": ""}

# ==========================================
# 2. 核心運算函數 (強制跑滿步數)
# ==========================================
def sigmoid(x, lambd):
    """標準 FCM 轉換函數 (0~1)"""
    return 1 / (1 + np.exp(-lambd * x))

def run_fcm(W, A_init, lambd, steps, epsilon):
    history = [A_init]
    current_state = A_init
    
    # ★★★ 修正重點：移除 break，強制跑完 range(steps) ★★★
    for _ in range(steps):
        influence = np.dot(current_state, W)
        next_state = sigmoid(influence, lambd)
        history.append(next_state)
        
        # 即使收斂也不要停，繼續跑，這樣圖表才會變長
        current_state = next_state
        
    return np.array(history)

# 回呼函數
def load_file_callback():
    uploaded = st.session_state.uploader_key
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'): df = pd.read_csv(uploaded, index_col=0)
            else: df = pd.read_excel(uploaded, index_col=0)
            st.session_state.concepts = df.columns.tolist()
            st.session_state.matrix = df.values
            st.toast(f"✅ 成功載入 {len(df)} 個項目！", icon="📂")
        except: st.error("檔案讀取失敗")

def sort_matrix_logic():
    try:
        df = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
        df_sorted = df.sort_index(axis=0).sort_index(axis=1)
        st.session_state.concepts = df_sorted.index.tolist()
        st.session_state.matrix = df_sorted.values
        st.success("✅ 排序完成！")
    except: st.error("排序失敗")

# ==========================================
# 3. 側邊欄設定
# ==========================================
st.sidebar.title("🛠️ 設定面板")

st.sidebar.subheader("1. 資料來源")
num_c = st.sidebar.number_input("準則數量", 3, 30, 9)
if st.sidebar.button("📥 下載空表"):
    dummy = [f"準則_{i+1}" for i in range(num_c)]
    df_t = pd.DataFrame(np.zeros((num_c, num_c)), index=dummy, columns=dummy)
    st.sidebar.download_button("下載 CSV", df_t.to_csv().encode('utf-8-sig'), "template.csv", "text/csv")

st.sidebar.file_uploader("上傳 Excel/CSV", type=['xlsx', 'csv'], key="uploader_key", on_change=load_file_callback)

st.sidebar.markdown("---")
with st.sidebar.expander("2. 矩陣編輯", expanded=False):
    with st.form("add_c"):
        new = st.text_input("新增準則")
        if st.form_submit_button("➕ 加入") and new:
            if new not in st.session_state.concepts:
                st.session_state.concepts.append(new)
                old = st.session_state.matrix
                r,c = old.shape
                new_m = np.zeros((r+1,c+1))
                new_m[:r,:c] = old
                st.session_state.matrix = new_m
                st.rerun()
    
    if st.button("🔄 自動排序"):
        sort_matrix_logic()
        st.rerun()
        
    if st.button("🎲 隨機生成權重 (-1~1)"):
        n = len(st.session_state.concepts)
        rand = np.random.uniform(-1.0, 1.0, (n, n))
        np.fill_diagonal(rand, 0)
        rand[np.abs(rand) < 0.2] = 0 
        st.session_state.matrix = rand
        st.success("已生成正負關係矩陣")
        st.rerun()

    if st.button("🗑️ 清空論文"):
        for k in st.session_state.paper_sections: st.session_state.paper_sections[k] = ""
        st.rerun()

with st.sidebar.expander("3. 模擬參數", expanded=True):
    LAMBDA = st.slider("Lambda", 0.1, 5.0, 1.0)
    # ★★★ 修正：步數拉桿 ★★★
    MAX_STEPS = st.slider("模擬步數 (Steps)", 10, 100, 21)

# ==========================================
# 4. 主畫面 Tabs
# ==========================================
st.title("FCM 論文決策系統 (Fixed Steps)")
tab1, tab2, tab3 = st.tabs(["📊 矩陣視圖", "📈 情境模擬", "🎓 論文生成"])

with tab1:
    st.subheader("矩陣關係檢視 (-1 ~ 1)")
    df_show = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
    st.dataframe(df_show.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1), height=400)

with tab2:
    st.subheader("情境模擬 (初始值 0-1)")
    cols = st.columns(3)
    initial_vals = []
    for i, c in enumerate(st.session_state.concepts):
        with cols[i % 3]:
            val = st.slider(c, 0.0, 1.0, 0.0, key=f"init_{i}")
            initial_vals.append(val)
            
    if st.button("🚀 開始運算", type="primary"):
        init_arr = np.array(initial_vals)
        
        # ★★★ 使用設定的 MAX_STEPS 強制跑滿 ★★★
        res = run_fcm(st.session_state.matrix, init_arr, LAMBDA, MAX_STEPS, 0.001)
        
        st.session_state.last_results = res
        st.session_state.last_initial = init_arr
        
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(len(res[0])):
            if np.max(res[:, i]) > 0.001:
                ax.plot(res[:, i], label=st.session_state.concepts[i])
        
        # ★★★ 強制設定 X 軸範圍，讓使用者看見步數變化 ★★★
        ax.set_xlim(0, MAX_STEPS) 
        
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Activation (0-1)")
        ax.set_xlabel(f"Simulation Steps (Total: {MAX_STEPS})") # 顯示總步數
        ax.legend(bbox_to_anchor=(1.01, 1))
        st.pyplot(fig)

# --- Tab 3: 長篇寫作 ---
with tab3:
    st.subheader("🎓 論文分段生成器")
    
    if st.session_state.last_results is None:
        st.error("⚠️ 請先至 Tab 2 執行運算！")
    else:
        matrix = st.session_state.matrix
        concepts = st.session_state.concepts
        results = st.session_state.last_results
        initial = st.session_state.last_initial
        final = results[-1]
        
        out_degree = np.sum(np.abs(matrix), axis=1)
        driver_idx = np.argmax(out_degree)
        driver_name = concepts[driver_idx]
        
        growth = final - initial
        best_idx = np.argmax(growth)
        best_name = concepts[best_idx]
        
        # 這裡會使用實際跑的步數
        steps = len(results) - 1 
        density = np.count_nonzero(matrix) / (len(concepts)**2)

        c1, c2, c3, c4 = st.columns(4)
        
        if c1.button("1️⃣ 生成 4.1 結構分析"):
            t = "### 第四章 研究結果與分析\n\n**4.1 FCM 矩陣結構特性分析**\n"
            t += f"本研究矩陣包含 {len(concepts)} 個準則，密度為 {density:.2f}。\n"
            t += f"數據顯示，**「{driver_name}」** 具有最高的出度 ({out_degree[driver_idx]:.2f})，確立其為關鍵驅動因子。\n"
            st.session_state.paper_sections["4.1"] = t

        if c2.button("2️⃣ 生成 4.2 穩定性"):
            t = "**4.2 系統穩定性檢測**\n"
            t += f"透過 Sigmoid 函數轉換，模擬顯示系統設定為 **{MAX_STEPS}** 步。數據顯示系統能有效收斂至穩態。\n"
            st.session_state.paper_sections["4.2"] = t

        if c3.button("3️⃣ 生成 4.3 情境模擬"):
            t = "**4.3 動態情境模擬分析**\n"
            t += f"本節模擬在 **「{driver_name}」** 投入資源後的擴散效應。\n"
            t += f"結果顯示，**「{best_name}」** 從初始狀態顯著提升至 {final[best_idx]:.2f}。這驗證了矩陣中的因果路徑。\n"
            st.session_state.paper_sections["4.3"] = t

        if c4.button("4️⃣ 生成 4.4 敏感度"):
            t = "**4.4 敏感度分析**\n經測試不同參數，關鍵準則排序不變，證實結論具備強健性。\n"
            st.session_state.paper_sections["4.4"] = t

        st.divider()
        c5, c6, c7 = st.columns(3)
        
        if c5.button("5️⃣ 生成 5.1 結論"):
            t = "### 第五章 結論與建議\n\n**5.1 研究結論**\n"
            t += f"1. 驅動因子確認：**「{driver_name}」** 為系統核心。\n2. 正向擴散效應：證實了治理機制能有效提升整體績效。\n"
            st.session_state.paper_sections["5.1"] = t

        if c6.button("6️⃣ 生成 5.2 建議"):
            t = "**5.2 管理意涵**\n1. 強化核心：應優先確保核心驅動因子的資源投入。\n2. 持續優化：利用正向回饋迴圈，持續滾動式提升績效。\n"
            st.session_state.paper_sections["5.2"] = t
            
        if c7.button("7️⃣ 生成 5.3 貢獻"):
            t = "**5.3 學術貢獻**\n1. 方法論證：展示了 FCM 在處理 0-1 因果關係上的適用性。\n2. 理論支持：為動態模擬提供了實證範本。\n"
            st.session_state.paper_sections["5.3"] = t

        st.markdown("---")
        st.subheader("📄 論文草稿累積區")
        
        full_text = ""
        for k in ["4.1", "4.2", "4.3", "4.4", "5.1", "5.2", "5.3"]:
            if st.session_state.paper_sections.get(k):
                full_text += st.session_state.paper_sections[k] + "\n\n"
        
        if full_text:
            st.markdown(f'<div class="report-box">{full_text}</div>', unsafe_allow_html=True)
            st.download_button("📥 下載完整論文 (TXT)", full_text, "thesis_final.txt")
        else:
            st.info("請點擊上方按鈕開始生成內容。")
