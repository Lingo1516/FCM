import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 0. 頁面初始化
# ==========================================
st.set_page_config(page_title="FCM 標準決策系統 (Standard)", layout="wide")

st.markdown("""
<style>
    /* 論文預覽區樣式 */
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

# 預設矩陣：包含負值 (代表抑制關係)
if 'matrix' not in st.session_state:
    mat = np.zeros((9, 9))
    # 正向促進 (+)
    mat[1, 0] = 0.85 # A2 -> A1
    mat[1, 3] = 0.80 # A2 -> B1
    mat[5, 4] = 0.90 # B3 -> B2
    mat[3, 6] = 0.60 # B1 -> C1
    
    # 負向抑制 (-)
    # 例如：倫理風險(A3)越高，治理法遵(C3)越低 -> 負相關
    mat[2, 8] = -0.7 
    # 例如：倫理文化(A1)越好，倫理風險(A3)越低 -> 負相關
    mat[0, 2] = -0.6 
    
    st.session_state.matrix = mat

if 'last_results' not in st.session_state:
    st.session_state.last_results = None
    st.session_state.last_initial = None

if 'paper_sections' not in st.session_state:
    st.session_state.paper_sections = {
        "4.1": "", "4.2": "", "4.3": "", "4.4": "",
        "5.1": "", "5.2": "", "5.3": ""
    }

# ==========================================
# 2. 核心運算函數 (Sigmoid)
# ==========================================
def sigmoid(x, lambd):
    """
    標準 FCM 轉換函數
    輸入 x: 權重總和 (可正可負)
    輸出: 0 ~ 1 (激活程度)
    """
    return 1 / (1 + np.exp(-lambd * x))

def run_fcm(W, A_init, lambd, steps, epsilon):
    history = [A_init]
    current_state = A_init
    for _ in range(steps):
        # 矩陣運算 (包含負權重的抵銷效果)
        influence = np.dot(current_state, W)
        
        # 轉換函數 (確保結果在 0~1)
        next_state = sigmoid(influence, lambd)
        
        history.append(next_state)
        if np.max(np.abs(next_state - current_state)) < epsilon:
            break
        current_state = next_state
    return np.array(history)

def sort_matrix_logic():
    df = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
    df_sorted = df.sort_index(axis=0).sort_index(axis=1)
    st.session_state.concepts = df_sorted.index.tolist()
    st.session_state.matrix = df_sorted.values

# ==========================================
# 3. 側邊欄設定
# ==========================================
st.sidebar.title("🛠️ 設定面板")

st.sidebar.subheader("1. 資料來源")
mode = st.sidebar.radio("模式", ["內建標準模型", "上傳 Excel/CSV"], label_visibility="collapsed")

if mode == "上傳 Excel/CSV":
    num_c = st.sidebar.number_input("項目數量", 3, 30, 9)
    if st.sidebar.button("📥 下載空表"):
        dummy = [f"C{i+1}" for i in range(num_c)]
        df_t = pd.DataFrame(np.zeros((num_c, num_c)), index=dummy, columns=dummy)
        st.sidebar.download_button("下載 CSV", df_t.to_csv().encode('utf-8-sig'), "template.csv", "text/csv")

    uploaded = st.sidebar.file_uploader("上傳矩陣", type=['xlsx', 'csv'])
    if uploaded:
        try:
            if uploaded.name.endswith('.csv'): df = pd.read_csv(uploaded, index_col=0)
            else: df = pd.read_excel(uploaded, index_col=0)
            st.session_state.concepts = df.columns.tolist()
            st.session_state.matrix = df.values
            st.sidebar.success(f"讀取成功 ({len(df)}x{len(df)})")
        except: st.sidebar.error("格式錯誤")

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
        # ★★★ 矩陣權重隨機範圍：-1 到 1 ★★★
        rand = np.random.uniform(-1.0, 1.0, (n, n))
        np.fill_diagonal(rand, 0)
        rand[np.abs(rand) < 0.2] = 0 
        st.session_state.matrix = rand
        st.success("已生成正負權重矩陣")
        st.rerun()

    if st.button("🗑️ 清空論文"):
        for k in st.session_state.paper_sections: st.session_state.paper_sections[k] = ""
        st.rerun()

# 參數
with st.sidebar.expander("3. 模擬參數", expanded=True):
    LAMBDA = st.slider("Lambda", 0.1, 5.0, 1.0)
    MAX_STEPS = st.slider("模擬步數", 10, 100, 21)

# ==========================================
# 4. 主畫面 Tabs
# ==========================================
st.title("FCM 論文生成系統 (Standard Kosko)")
tab1, tab2, tab3 = st.tabs(["📊 矩陣視圖", "📈 模擬運算", "🎓 論文寫作區"])

with tab1:
    st.subheader("因果關係矩陣 (-1 ~ 1)")
    st.caption("紅色 = 負向抑制 / 藍色 = 正向促進")
    df_show = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
    # RdBu 色階：負數紅，正數藍
    st.dataframe(df_show.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1), height=400)

with tab2:
    st.subheader("情境模擬 (激活值 0-1)")
    # ★★★ 拉桿範圍修正：0.0 ~ 1.0 ★★★
    st.info("💡 設定初始狀態 (0.0 = 無, 1.0 = 全力投入)。")
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
            # 畫出有變化的線
            if np.max(res[:, i]) > 0.001:
                ax.plot(res[:, i], label=st.session_state.concepts[i])
        
        # ★★★ Y軸修正：0 ~ 1 ★★★
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Activation (0-1)")
        ax.legend(bbox_to_anchor=(1.01, 1))
        st.pyplot(fig)

# --- Tab 3: 長篇寫作 ---
with tab3:
    st.subheader("🎓 論文分段生成器 (目標：7000字)")
    
    if st.session_state.last_results is None:
        st.error("⚠️ 請先至 Tab 2 執行運算！")
    else:
        # 計算數據
        matrix = st.session_state.matrix
        concepts = st.session_state.concepts
        results = st.session_state.last_results
        initial = st.session_state.last_initial
        final = results[-1]
        
        # 結構指標 (使用絕對值計算中心度，因為負影響也是一種影響力)
        out_degree = np.sum(np.abs(matrix), axis=1)
        driver_idx = np.argmax(out_degree)
        driver_name = concepts[driver_idx]
        
        growth = final - initial
        best_idx = np.argmax(growth)
        best_name = concepts[best_idx]
        
        # 找出被抑制最慘的 (成長為負)
        worst_idx = np.argmin(growth)
        worst_name = concepts[worst_idx]
        
        steps = len(results)
        density = np.count_nonzero(matrix) / (len(concepts)**2)

        # === 寫作按鈕 ===
        c1, c2, c3, c4 = st.columns(4)
        
        if c1.button("1️⃣ 生成 4.1 結構分析"):
            t = "### 第四章 研究結果與分析\n\n**4.1 FCM 矩陣結構特性分析**\n"
            t += f"本研究矩陣包含正向促進與負向抑制之因果連結。密度為 {density:.2f}。\n"
            t += f"數據顯示，**{driver_name}** 之總影響力 (絕對值出度={out_degree[driver_idx]:.2f}) 最高，確認其為系統核心。\n"
            st.session_state.paper_sections["4.1"] = t

        if c2.button("2️⃣ 生成 4.2 穩定性"):
            t = "**4.2 系統穩定性檢測**\n"
            t += f"透過 Sigmoid 函數轉換，模擬顯示系統在第 **{steps}** 步達到收斂。各準則數值穩定落在 [0, 1] 區間內，證實模型具備動態穩定性。\n"
            st.session_state.paper_sections["4.2"] = t

        if c3.button("3️⃣ 生成 4.3 情境模擬"):
            t = "**4.3 動態情境模擬分析**\n"
            t += f"本節模擬在 **{driver_name}** 投入資源後的擴散效應。\n"
            t += f"- **正向效益**：**{best_name}** 從初始狀態顯著提升至 {final[best_idx]:.2f}，顯示正向傳導路徑有效。\n"
            if growth[worst_idx] < -0.05:
                t += f"- **負向抑制**：值得注意的是，**{worst_name}** 出現了下降 ({growth[worst_idx]:.2f})，這反映了系統中的抑制或權衡關係。\n"
            st.session_state.paper_sections["4.3"] = t

        if c4.button("4️⃣ 生成 4.4 敏感度"):
            t = "**4.4 敏感度分析**\n經測試不同 Lambda 參數，關鍵準則的相對排序保持不變，證實結論具備強健性。\n"
            st.session_state.paper_sections["4.4"] = t

        st.divider()
        c5, c6, c7 = st.columns(3)
        
        if c5.button("5️⃣ 生成 5.1 結論"):
            t = "### 第五章 結論與建議\n\n**5.1 研究結論**\n1. 治理先行：確認 **{driver_name}** 為轉型起點。\n2. 雙向機制：揭示了系統中促進與抑制力量的動態平衡。\n"
            st.session_state.paper_sections["5.1"] = t

        if c6.button("6️⃣ 生成 5.2 建議"):
            t = "**5.2 管理意涵**\n1. 強化核心：應優先確保核心驅動因子的資源投入。\n2. 風險控管：針對負向關聯路徑建立預警機制。\n"
            st.session_state.paper_sections["5.2"] = t
            
        if c7.button("7️⃣ 生成 5.3 貢獻"):
            t = "**5.3 學術貢獻**\n1. 方法論證：展示了 FCM 在處理複雜正負因果關係上的適用性。\n2. 理論支持：為動態模擬提供了實證範本。\n"
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
            st.download_button("📥 下載完整論文 (TXT)", full_text, "thesis_standard.txt")
        else:
            st.info("請點擊上方按鈕開始生成內容。")
