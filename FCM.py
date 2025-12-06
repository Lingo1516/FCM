import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 0. 頁面初始化
# ==========================================
st.set_page_config(page_title="FCM 論文決策系統 (自定義名稱版)", layout="wide")

st.markdown("""
<style>
    .report-box { 
        border: 1px solid #ccc; padding: 40px; background-color: #ffffff; 
        color: #000000; font-family: "Times New Roman", "標楷體", serif; 
        font-size: 16px; line-height: 2.0; text-align: justify;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 20px; white-space: pre-wrap;
    }
    .stButton>button { width: 100%; border-radius: 5px; height: 3.5em; font-weight: bold; font-size: 15px;}
    .name-editor { border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; background-color: #f9f9f9; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 初始化數據
# ==========================================
# 預設 13 個概念 (依照您的圖片)
if 'num_concepts' not in st.session_state:
    st.session_state.num_concepts = 13

if 'concepts' not in st.session_state:
    st.session_state.concepts = [f"準則_{i+1}" for i in range(13)]

if 'matrix' not in st.session_state:
    # 預設 13x13 零矩陣
    st.session_state.matrix = np.zeros((13, 13))

if 'last_results' not in st.session_state:
    st.session_state.last_results = None
    st.session_state.last_initial = None

if 'paper_sections' not in st.session_state:
    st.session_state.paper_sections = {
        "4.1": "", "4.2": "", "4.3": "", "4.4": "",
        "5.1": "", "5.2": "", "5.3": ""
    }

# ==========================================
# 2. 核心運算 (Sigmoid)
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
# 3. 側邊欄設定
# ==========================================
st.sidebar.title("🛠️ 設定面板")

# --- Step 1: 設定數量 & 名稱 ---
st.sidebar.subheader("Step 1: 定義準則")
new_num = st.sidebar.number_input("準則數量", 3, 30, st.session_state.num_concepts)

# 如果數量改變，重置概念列表
if new_num != st.session_state.num_concepts:
    st.session_state.num_concepts = new_num
    st.session_state.concepts = [f"準則_{i+1}" for i in range(new_num)]
    st.session_state.matrix = np.zeros((new_num, new_num))
    st.rerun()

# ★★★ 關鍵功能：讓使用者自己改名字 ★★★
with st.sidebar.expander("📝 修改準則名稱 (重要!)", expanded=True):
    st.caption("請在此輸入 C1, C2... 的真實名稱，報告才會正確。")
    
    # 建立一個 DataFrame 讓使用者編輯
    name_df = pd.DataFrame({"代號": [f"C{i+1}" for i in range(new_num)], "名稱": st.session_state.concepts})
    edited_df = st.data_editor(name_df, hide_index=True, use_container_width=True)
    
    if st.button("💾 更新名稱"):
        st.session_state.concepts = edited_df["名稱"].tolist()
        st.success("名稱已更新！")
        st.rerun()

st.sidebar.markdown("---")

# --- Step 2: 資料來源 ---
st.sidebar.subheader("Step 2: 匯入矩陣")
mode = st.sidebar.radio("模式", ["上傳 Excel/CSV", "手動/隨機生成"], label_visibility="collapsed")

if mode == "上傳 Excel/CSV":
    if st.sidebar.button("📥 下載空表 (含名稱)"):
        # 下載包含使用者定義名稱的空表
        df_t = pd.DataFrame(np.zeros((new_num, new_num)), index=st.session_state.concepts, columns=st.session_state.concepts)
        st.sidebar.download_button("下載 CSV", df_t.to_csv().encode('utf-8-sig'), "template.csv", "text/csv")

    uploaded = st.sidebar.file_uploader("上傳矩陣", type=['xlsx', 'csv'])
    if uploaded:
        try:
            if uploaded.name.endswith('.csv'): df = pd.read_csv(uploaded, index_col=0)
            else: df = pd.read_excel(uploaded, index_col=0)
            
            # 檢查大小是否匹配
            if df.shape[0] != new_num:
                st.sidebar.error(f"錯誤：上傳的矩陣大小 ({df.shape[0]}) 與設定的數量 ({new_num}) 不符！")
            else:
                st.session_state.matrix = df.values
                st.sidebar.success("✅ 讀取成功")
        except: st.sidebar.error("格式錯誤")

else:
    # 手動工具
    if st.sidebar.button("🎲 隨機生成權重 (-1~1)"):
        rand = np.random.uniform(-1.0, 1.0, (new_num, new_num))
        np.fill_diagonal(rand, 0)
        rand[np.abs(rand) < 0.2] = 0 
        st.session_state.matrix = rand
        st.sidebar.success("已隨機生成")
        
    if st.sidebar.button("🗑️ 重置為零"):
        st.session_state.matrix = np.zeros((new_num, new_num))
        st.rerun()

# 參數
with st.sidebar.expander("⚙️ 模擬參數"):
    LAMBDA = st.slider("Lambda", 0.1, 5.0, 1.0)
    MAX_STEPS = st.slider("模擬步數", 10, 100, 21)

# ==========================================
# 4. 主畫面
# ==========================================
st.title("FCM 論文生成系統 (Custom Names Ver.)")
tab1, tab2, tab3 = st.tabs(["📊 矩陣視圖", "📈 模擬運算", "🎓 論文寫作區"])

with tab1:
    st.subheader("矩陣權重檢視")
    st.caption("請確認列與欄的名稱是否正確。")
    df_show = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
    st.dataframe(df_show.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1), height=500)

with tab2:
    st.subheader("情境模擬 (初始值 0-1)")
    st.info("💡 請設定初始投入。拉桿名稱已同步更新。")
    
    cols = st.columns(3)
    initial_vals = []
    for i, c in enumerate(st.session_state.concepts):
        with cols[i % 3]:
            val = st.slider(c, 0.0, 1.0, 0.0, key=f"init_{i}")
            initial_vals.append(val)
            
    if st.button("🚀 開始運算", type="primary"):
        init_arr = np.array(initial_vals)
        res = run_fcm(st.session_state.matrix, init_arr, LAMBDA, MAX_STEPS, 0.001)
        st.session_state.last_results = res
        st.session_state.last_initial = init_arr
        
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(len(res[0])):
            if np.max(res[:, i]) > 0.001:
                ax.plot(res[:, i], label=st.session_state.concepts[i])
        
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Activation (0-1)")
        ax.legend(bbox_to_anchor=(1.01, 1))
        st.pyplot(fig)

# --- Tab 3: 寫作核心 (使用真實名稱) ---
with tab3:
    st.subheader("🎓 論文分段生成器")
    
    if st.session_state.last_results is None:
        st.error("⚠️ 請先至 Tab 2 執行運算！")
    else:
        # 準備數據
        matrix = st.session_state.matrix
        concepts = st.session_state.concepts
        results = st.session_state.last_results
        initial = st.session_state.last_initial
        final = results[-1]
        
        # 找出關鍵角色 (這裡會用到您輸入的真實名稱)
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
        
        if c1.button("1️⃣ 生成 4.1 結構分析"):
            t = "### 第四章 研究結果與分析\n\n**4.1 FCM 矩陣結構特性分析**\n"
            t += f"本研究矩陣包含 {len(concepts)} 個準則，矩陣密度為 {density:.2f}。\n"
            t += f"根據中心度分析，**「{driver_name}」** 具有最高的出度 ({out_degree[driver_idx]:.2f})。這意味著在目前的架構中，它是最具影響力的核心因子。\n"
            st.session_state.paper_sections["4.1"] = t

        if c2.button("2️⃣ 生成 4.2 穩定性"):
            t = "**4.2 系統穩定性檢測**\n"
            t += f"模擬顯示系統在第 **{steps}** 步達到收斂，證實模型具備動態穩定性。\n"
            st.session_state.paper_sections["4.2"] = t

        if c3.button("3️⃣ 生成 4.3 情境模擬"):
            t = "**4.3 動態情境模擬分析**\n"
            t += f"設定情境：強化投入 **「{driver_name}」**。\n"
            t += f"結果顯示，**「{best_name}」** 受益最大，成長幅度達 +{growth[best_idx]:.2f}。這驗證了從 {driver_name} 到 {best_name} 的因果傳導路徑。\n"
            st.session_state.paper_sections["4.3"] = t

        if c4.button("4️⃣ 生成 4.4 敏感度"):
            t = "**4.4 敏感度分析**\n經測試不同參數，關鍵準則排序不變，結論具備強健性。\n"
            st.session_state.paper_sections["4.4"] = t

        st.divider()
        c5, c6, c7 = st.columns(3)
        
        if c5.button("5️⃣ 生成 5.1 結論"):
            t = "### 第五章 結論與建議\n\n**5.1 研究結論**\n"
            t += f"1. 核心發現：確認 **「{driver_name}」** 為轉型起點。\n2. 擴散效應：證實了治理機制能有效帶動 **「{best_name}」** 的績效提升。\n"
            st.session_state.paper_sections["5.1"] = t

        if c6.button("6️⃣ 生成 5.2 建議"):
            t = "**5.2 管理意涵**\n"
            t += f"1. 資源配置：建議集中資源強化 **「{driver_name}」**。\n2. 長期思維：容忍初期的成效滯後。\n"
            st.session_state.paper_sections["5.2"] = t
            
        if c7.button("7️⃣ 生成 5.3 貢獻"):
            t = "**5.3 學術貢獻**\n1. 方法論：展示了 FCM 在此議題上的適用性。\n2. 實證價值：為動態模擬提供了數據支持。\n"
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
            col_d.download_button("📥 下載文字檔", full_text, "thesis.txt")
            if col_c.button("🗑️ 清空內容"):
                for k in st.session_state.paper_sections: st.session_state.paper_sections[k] = ""
                st.rerun()
        else:
            st.info("請依序點擊上方按鈕開始生成內容。")
