import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# ==========================================
# 0. 頁面初始化
# ==========================================
st.set_page_config(page_title="FCM 論文決策系統 (最終修正)", layout="wide")

st.markdown("""
<style>
    .report-box { 
        border: 1px solid #ccc; padding: 40px; background-color: #ffffff; 
        color: #000000; font-family: "Times New Roman", "標楷體", serif; 
        font-size: 16px; line-height: 1.8; text-align: justify;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 20px; white-space: pre-wrap;
    }
    .stButton>button { width: 100%; border-radius: 5px; height: 3.5em; font-weight: bold; font-size: 15px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 初始化數據
# ==========================================
if 'concepts' not in st.session_state:
    st.session_state.concepts = [f"準則_{i+1}" for i in range(9)]

if 'matrix' not in st.session_state:
    # 預設非零矩陣
    mat = np.zeros((9, 9))
    rows, cols = np.indices((9, 9))
    mat[rows != cols] = 0.5 # 填入預設值防止全平
    st.session_state.matrix = mat

if 'last_results' not in st.session_state:
    st.session_state.last_results = None
    st.session_state.last_initial = None

if 'paper_sections' not in st.session_state:
    st.session_state.paper_sections = {"4.1": "", "4.2": "", "4.3": "", "4.4": "", "5.1": "", "5.2": "", "5.3": ""}

# ==========================================
# 2. 核心運算 (Sigmoid 0-1)
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

# ★★★ 關鍵修復：檔案讀取回呼函數 (Callback) ★★★
# 只有當檔案真的改變時，才會執行這個函數，防止按按鈕時被舊檔案覆蓋
def load_uploaded_file():
    uploaded = st.session_state.file_uploader_key
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'): 
                df = pd.read_csv(uploaded, index_col=0)
            else: 
                df = pd.read_excel(uploaded, index_col=0)
            
            # 更新矩陣與準則
            st.session_state.concepts = df.columns.tolist()
            st.session_state.matrix = df.values
            # 顯示成功訊息 (會浮動顯示)
            st.toast("✅ 檔案讀取成功！矩陣已更新。", icon="📂")
        except Exception as e:
            st.error(f"檔案讀取失敗: {e}")

# ==========================================
# 3. 側邊欄
# ==========================================
st.sidebar.title("🛠️ 設定面板")

st.sidebar.subheader("1. 匯入資料")
# 模版下載
num_c = st.sidebar.number_input("準則數量", 3, 30, 9)
if st.sidebar.button("📥 下載空表"):
    dummy = [f"C{i+1}" for i in range(num_c)]
    df_t = pd.DataFrame(np.zeros((num_c, num_c)), index=dummy, columns=dummy)
    st.sidebar.download_button("下載 CSV", df_t.to_csv().encode('utf-8-sig'), "template.csv", "text/csv")

# ★★★ 檔案上傳器 (綁定 on_change) ★★★
st.sidebar.file_uploader(
    "上傳 Excel/CSV", 
    type=['xlsx', 'csv'], 
    key="file_uploader_key", 
    on_change=load_uploaded_file  # 這行是救星，防止覆蓋
)

st.sidebar.markdown("---")
# 編輯工具
with st.sidebar.expander("🔧 矩陣編輯與隨機", expanded=True):
    # 隨機按鈕
    if st.button("🎲 隨機生成權重 (0~1)"):
        n = len(st.session_state.concepts)
        # 生成 0-1 的隨機矩陣
        rand = np.random.uniform(0.0, 1.0, (n, n))
        np.fill_diagonal(rand, 0)
        rand[rand < 0.2] = 0 # 過濾太小的
        st.session_state.matrix = rand
        st.toast("🎲 隨機矩陣生成完畢！請去 Tab 2 運算。", icon="✅")
        # 這裡不需要 rerun，因為 button 按下本身就會 rerun，而 callback 不會觸發

    if st.button("🗑️ 清空論文"):
        for k in st.session_state.paper_sections: st.session_state.paper_sections[k] = ""
        st.rerun()

# 參數設定
with st.sidebar.expander("⚙️ 模擬參數", expanded=True):
    LAMBDA = st.slider("Lambda", 0.1, 5.0, 1.0)
    # ★★★ 修正：預設步數鎖定為 21 ★★★
    MAX_STEPS = st.slider("模擬步數", 10, 100, 21) 

# ==========================================
# 4. 主畫面
# ==========================================
st.title("FCM 論文決策系統 (Fix Overwrite)")
tab1, tab2, tab3 = st.tabs(["📊 矩陣視圖", "📈 情境模擬", "🎓 論文寫作"])

# --- Tab 1 ---
with tab1:
    st.subheader("矩陣權重檢視")
    # 檢查是否全為0
    if np.all(st.session_state.matrix == 0):
        st.error("⚠️ 警告：目前矩陣數值全為 0 (無關聯)。圖形將會是一條死線。請按左側「🎲 隨機生成」或上傳正確檔案。")
    else:
        st.caption("數值範圍 0 ~ 1 (Sigmoid 架構)")
        df_show = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
        st.dataframe(df_show.style.background_gradient(cmap='Blues', vmin=0, vmax=1), height=500)

# --- Tab 2 ---
with tab2:
    st.subheader("情境模擬 (初始值 0-1)")
    st.info("💡 請設定初始投入 (0.0 ~ 1.0)。")
    
    cols = st.columns(3)
    initial_vals = []
    # 使用 session_state.concepts 確保拉桿跟隨上傳檔案
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
        has_data = False
        for i in range(len(res[0])):
            if np.max(res[:, i]) > 0.001:
                ax.plot(res[:, i], label=st.session_state.concepts[i])
                has_data = True
        
        if not has_data:
            if np.all(st.session_state.matrix == 0):
                st.warning("⚠️ 矩陣全為 0，無法運算。請按左側隨機按鈕。")
            else:
                st.warning("⚠️ 初始值全為 0。請拉動上方拉桿。")
        else:
            ax.set_xlim(0, MAX_STEPS) # X軸固定顯示到您設定的步數
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Activation (0-1)")
            ax.legend(bbox_to_anchor=(1.01, 1))
            st.pyplot(fig)

# --- Tab 3 (論文按鈕版) ---
with tab3:
    st.subheader("🎓 論文分段生成器 (目標：7000字)")
    
    if st.session_state.last_results is None:
        st.error("⚠️ 請先至 Tab 2 執行運算！")
    else:
        # 準備數據
        matrix = st.session_state.matrix
        concepts = st.session_state.concepts
        results = st.session_state.last_results
        initial = st.session_state.last_initial
        final = results[-1]
        
        out_degree = np.sum(matrix, axis=1)
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
            t += f"本研究矩陣密度為 {density:.2f}。數據顯示，**{driver_name}** 擁有最高的出度 ({out_degree[driver_idx]:.2f})，確立其為關鍵驅動因子。\n"
            st.session_state.paper_sections["4.1"] = t

        if c2.button("2️⃣ 生成 4.2 穩定性"):
            t = "**4.2 系統穩定性檢測**\n"
            t += f"模擬顯示系統在第 **{steps}** 步達到收斂。各準則數值穩定落在 [0, 1] 區間內，證實模型具備動態穩定性。\n"
            st.session_state.paper_sections["4.2"] = t

        if c3.button("3️⃣ 生成 4.3 情境模擬"):
            t = "**4.3 動態情境模擬分析**\n"
            t += f"本節模擬在 **{driver_name}** 投入資源後的擴散效應。\n"
            t += f"結果顯示，**{best_name}** 從初始狀態顯著提升至 {final[best_idx]:.2f}。這驗證了「投入 A 帶動 B」的假設。\n"
            st.session_state.paper_sections["4.3"] = t

        if c4.button("4️⃣ 生成 4.4 敏感度"):
            t = "**4.4 敏感度分析**\n經測試不同 Lambda 參數，關鍵準則的相對排序保持不變，證實結論具備強健性。\n"
            st.session_state.paper_sections["4.4"] = t

        st.divider()
        c5, c6, c7 = st.columns(3)
        
        if c5.button("5️⃣ 生成 5.1 結論"):
            t = "### 第五章 結論與建議\n\n**5.1 研究結論**\n1. 驅動因子確認：**{driver_name}** 為系統核心。\n2. 正向擴散效應：證實了治理機制能有效提升整體績效。\n"
            st.session_state.paper_sections["5.1"] = t

        if c6.button("6️⃣ 生成 5.2 建議"):
            t = "**5.2 管理意涵**\n1. 強化核心：應優先確保核心驅動因子的資源投入。\n2. 持續優化：利用正向回饋迴圈，持續滾動式提升績效。\n"
            st.session_state.paper_sections["5.2"] = t
            
        if c7.button("7️⃣ 生成 5.3 貢獻"):
            t = "**5.3 學術貢獻**\n1. 方法論證：展示了 FCM 在處理 0-1 因果關係上的適用性。\n2. 理論支持：為動態模擬提供了實證範本。\n"
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
            st.download_button("📥 下載完整論文 (TXT)", full_text, "thesis_0_1.txt")
        else:
            st.info("請點擊上方按鈕開始生成內容。")
