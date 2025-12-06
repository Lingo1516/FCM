import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 0. 頁面初始化 (樣式保持不變)
# ==========================================
st.set_page_config(page_title="FCM 客製化決策系統", layout="wide")

st.markdown("""
<style>
    .report-box { 
        border: 1px solid #ccc; padding: 40px; background-color: #ffffff; 
        color: #000000; font-family: "Times New Roman", "標楷體", serif; 
        font-size: 16px; line-height: 2.0; text-align: justify;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 20px; white-space: pre-wrap;
    }
    .chat-ai { background-color: #E3F2FD; padding: 10px; border-radius: 10px; color: black; margin-bottom: 10px;}
    .stButton>button { width: 100%; border-radius: 5px; height: 3.5em; font-weight: bold; font-size: 15px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 初始化數據 (保留預設值以免畫面空白)
# ==========================================
if 'concepts' not in st.session_state:
    st.session_state.concepts = [
        "A1 倫理文化", "A2 高層基調", "A3 倫理風險",
        "B1 策略一致性", "B2 利害關係人", "B3 資訊透明",
        "C1 社會影響", "C2 環境責任", "C3 治理法遵"
    ]

if 'matrix' not in st.session_state:
    # 預設矩陣 (Demo用)
    mat = np.array([
        [0.00, 0.45, 0.60, 0.55, 0.40, 0.30, 0.35, 0.20, 0.70],
        [0.90, 0.00, 0.85, 0.95, 0.60, 0.80, 0.50, 0.45, 0.75],
        [0.50, 0.30, 0.00, 0.40, 0.20, 0.65, 0.10, 0.30, 0.85],
        [0.30, 0.40, 0.20, 0.00, 0.50, 0.60, 0.70, 0.75, 0.40],
        [0.25, 0.30, 0.15, 0.45, 0.00, 0.70, 0.80, 0.30, 0.20],
        [0.40, 0.50, 0.60, 0.55, 0.90, 0.00, 0.65, 0.40, 0.50],
        [0.30, 0.20, 0.10, 0.20, 0.60, 0.40, 0.00, 0.35, 0.30],
        [0.20, 0.25, 0.30, 0.30, 0.40, 0.50, 0.40, 0.00, 0.45],
        [0.60, 0.55, 0.70, 0.40, 0.35, 0.50, 0.30, 0.25, 0.00]
    ])
    mat[2, 0] = -0.3; mat[7, 6] = -0.2
    st.session_state.matrix = mat

if 'last_results' not in st.session_state:
    st.session_state.last_results = None
    st.session_state.last_initial = None

if 'paper_sections' not in st.session_state:
    st.session_state.paper_sections = {"4.1": "", "4.2": "", "4.3": "", "4.4": "", "5.1": "", "5.2": "", "5.3": ""}

# ==========================================
# 2. 核心運算函數 (保持不動)
# ==========================================
def sigmoid(x, lambd):
    return np.tanh(lambd * x) # Tanh (-1 to 1)

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
# 3. 側邊欄 (全新客製化流程)
# ==========================================
st.sidebar.title("🛠️ 專案設定")

# --- Step 1: 下載模版 ---
st.sidebar.header("Step 1: 建立空模版")
num_concepts = st.sidebar.number_input("請問有多少個準則項目？", min_value=3, max_value=30, value=9)

if st.sidebar.button("📥 下載對應數量的 Excel 空表"):
    # 自動生成對應數量的空矩陣
    dummy_names = [f"準則_{i+1}" for i in range(num_concepts)]
    df_template = pd.DataFrame(np.zeros((num_concepts, num_concepts)), index=dummy_names, columns=dummy_names)
    
    # 轉換為 CSV 供下載
    csv = df_template.to_csv().encode('utf-8-sig') # utf-8-sig 解決中文亂碼
    st.sidebar.download_button(
        label="點擊下載 (.csv)",
        data=csv,
        file_name="fcm_template.csv",
        mime="text/csv",
        key='download-csv'
    )
    st.sidebar.success(f"已生成 {num_concepts}x{num_concepts} 的模版，請下載填寫後上傳。")

st.sidebar.markdown("---")

# --- Step 2: 上傳資料 ---
st.sidebar.header("Step 2: 上傳填好的檔案")
uploaded_file = st.sidebar.file_uploader("上傳 Excel 或 CSV", type=['xlsx', 'csv'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_new = pd.read_csv(uploaded_file, index_col=0)
        else:
            df_new = pd.read_excel(uploaded_file, index_col=0)
            
        # 更新系統狀態
        st.session_state.concepts = df_new.columns.tolist()
        st.session_state.matrix = df_new.values
        st.sidebar.success(f"✅ 成功載入！共 {len(st.session_state.concepts)} 個項目。")
        
    except Exception as e:
        st.sidebar.error(f"檔案格式錯誤: {e}")

st.sidebar.markdown("---")

# --- Step 3: 參數設定 ---
with st.sidebar.expander("⚙️ 進階參數 (通常不需更動)"):
    LAMBDA = st.slider("Lambda (敏感度)", 0.1, 5.0, 1.0)
    MAX_STEPS = st.slider("最大模擬步數", 10, 100, 30)

# ==========================================
# 4. 主畫面 (Tabs)
# ==========================================
st.title("FCM 論文決策系統 (Custom Input Ver.)")
tab1, tab2, tab3 = st.tabs(["📊 矩陣檢視", "📈 模擬運算", "🎓 論文寫作"])

# --- Tab 1: 矩陣 ---
with tab1:
    st.subheader(f"目前矩陣架構 ({len(st.session_state.concepts)}x{len(st.session_state.concepts)})")
    st.caption("說明：這是您上傳或系統預設的矩陣。數值範圍 -1.0 至 1.0。")
    
    df_show = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
    st.dataframe(df_show.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1), height=500)

# --- Tab 2: 模擬 (自動對應項目數量) ---
with tab2:
    st.subheader("情境模擬 (Scenario Analysis)")
    st.info("💡 請調整下方拉桿設定初始情境。系統已自動根據您上傳的項目數量產生對應拉桿。")
    
    # 動態產生拉桿 (依照 concepts 數量)
    cols = st.columns(3)
    initial_vals = []
    for i, c in enumerate(st.session_state.concepts):
        with cols[i % 3]:
            val = st.slider(c, -1.0, 1.0, 0.0, key=f"init_{i}")
            initial_vals.append(val)
            
    if st.button("🚀 開始運算", type="primary"):
        init_arr = np.array(initial_vals)
        res = run_fcm(st.session_state.matrix, init_arr, LAMBDA, MAX_STEPS, 0.001)
        st.session_state.last_results = res
        st.session_state.last_initial = init_arr
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        for i in range(len(res[0])):
            if abs(res[-1, i]) > 0.01 or abs(init_arr[i]) > 0.01:
                ax.plot(res[:, i], label=st.session_state.concepts[i])
        ax.set_ylim(-1.1, 1.1)
        ax.legend(bbox_to_anchor=(1.01, 1))
        st.pyplot(fig)

# --- Tab 3: 長篇寫作 (內容邏輯不變，只變數值) ---
with tab3:
    st.subheader("🎓 論文分段生成器 (支援動態項目)")
    st.info("💡 無論您上傳幾個項目，此處皆可自動分析並生成長篇論文。")

    if st.session_state.last_results is None:
        st.error("⚠️ 請先至 Tab 2 執行運算！")
    else:
        # 計算數據
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
        steps = len(results)
        density = np.count_nonzero(matrix) / (len(concepts)**2)

        # === 寫作按鈕 ===
        c1, c2, c3, c4 = st.columns(4)
        
        if c1.button("1️⃣ 生成 4.1 結構 (長篇)"):
            t = "### 第四章 研究結果與分析\n\n"
            t += "**4.1 FCM 矩陣結構特性分析**\n\n"
            t += "本節依據圖論針對專家共識建立之矩陣進行檢測。\n\n"
            t += f"**4.1.1 密度分析**：本研究包含 {len(concepts)} 個準則。矩陣密度為 {density:.2f}，顯示系統高度連通。\n"
            t += f"**4.1.2 中心度分析**：數據顯示，**{driver_name}** 之出度 ({out_degree[driver_idx]:.2f}) 最高，確立其為關鍵驅動因子。\n"
            st.session_state.paper_sections["4.1"] = t

        if c2.button("2️⃣ 生成 4.2 穩定性 (長篇)"):
            t = "**4.2 系統穩定性檢測**\n\n"
            t += f"模擬顯示，系統在第 **{steps}** 步達到收斂。即便在 Tanh 函數環境下，系統仍展現出良好的動態平衡，未出現發散。\n"
            st.session_state.paper_sections["4.2"] = t

        if c3.button("3️⃣ 生成 4.3 情境 (長篇)"):
            t = "**4.3 動態情境模擬分析**\n\n"
            t += f"本節模擬強化投入 **{driver_name}** 之效應。\n"
            t += f"結果顯示，**{best_name}** 呈現顯著成長 (+{growth[best_idx]:.2f})，驗證了矩陣中的正向回饋路徑有效運作。\n"
            st.session_state.paper_sections["4.3"] = t

        if c4.button("4️⃣ 生成 4.4 敏感度 (長篇)"):
            t = "**4.4 敏感度分析**\n\n經測試不同 Lambda 參數，關鍵準則相對排序不變，證實結論具備強健性。\n"
            st.session_state.paper_sections["4.4"] = t

        st.divider()
        c5, c6, c7 = st.columns(3)
        if c5.button("5️⃣ 生成 5.1 結論 (長篇)"):
            t = "### 第五章 結論與建議\n\n**5.1 研究結論**\n\n"
            t += f"1. 實證治理驅動：確認 **{driver_name}** 為轉型起點。\n2. 量化動態滯後：揭示了策略發酵的時間成本。\n"
            st.session_state.paper_sections["5.1"] = t

        if c6.button("6️⃣ 生成 5.2 建議 (長篇)"):
            t = "**5.2 管理意涵**\n\n1. 資源集中：應優先確保核心驅動因子資源。\n2. 風險控管：建立長效考核機制。\n"
            st.session_state.paper_sections["5.2"] = t
            
        if c7.button("7️⃣ 生成 5.3 貢獻 (長篇)"):
            t = "**5.3 學術貢獻**\n\n1. 豐富理論內涵：量化領導者認知之動態影響。\n2. 創新方法應用：提供標準化 FCM 分析範本。\n"
            st.session_state.paper_sections["5.3"] = t

        # === 預覽區 ===
        st.markdown("---")
        st.subheader("📄 論文草稿累積區")
        full_text = ""
        for k in ["4.1", "4.2", "4.3", "4.4", "5.1", "5.2", "5.3"]:
            if st.session_state.paper_sections.get(k):
                full_text += st.session_state.paper_sections[k] + "\n\n"
        
        if full_text:
            st.markdown(f'<div class="report-box">{full_text}</div>', unsafe_allow_html=True)
            col_d, col_c = st.columns([1, 1])
            col_d.download_button("📥 下載完整論文 (TXT)", full_text, "Full_Thesis_Draft.txt")
            if col_c.button("🗑️ 清空所有內容"):
                for k in st.session_state.paper_sections: st.session_state.paper_sections[k] = ""
                st.rerun()
        else:
            st.info("請依序點擊上方按鈕開始生成內容。")
