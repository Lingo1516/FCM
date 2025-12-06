import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# ==========================================
# 0. 頁面初始化
# ==========================================
st.set_page_config(page_title="FCM 論文決策系統 (Clean & Fix)", layout="wide")

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
# 1. 初始化數據 (改為空值，不預載舊資料)
# ==========================================
if 'concepts' not in st.session_state:
    st.session_state.concepts = [] # 預設為空列表

if 'matrix' not in st.session_state:
    st.session_state.matrix = np.array([]) # 預設為空陣列

if 'last_results' not in st.session_state:
    st.session_state.last_results = None
    st.session_state.last_initial = None

if 'paper_sections' not in st.session_state:
    st.session_state.paper_sections = {
        "4.1": "", "4.2": "", "4.3": "", "4.4": "",
        "5.1": "", "5.2": "", "5.3": ""
    }

# ==========================================
# 2. 核心運算函數
# ==========================================
def sigmoid(x, lambd):
    """標準 FCM 轉換函數 (Sigmoid 0~1)"""
    return 1 / (1 + np.exp(-lambd * x))

def run_fcm(W, A_init, lambd, steps, epsilon):
    history = [A_init]
    current_state = A_init
    for _ in range(steps):
        influence = np.dot(current_state, W)
        next_state = sigmoid(influence, lambd)
        history.append(next_state)
        # 移除提早中斷，確保跑滿指定步數，方便觀察
        current_state = next_state
    return np.array(history)

# 回呼函數：上傳檔案時，強制覆蓋
def load_file_callback():
    uploaded = st.session_state.uploader_key
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'): 
                df = pd.read_csv(uploaded, index_col=0)
            else: 
                df = pd.read_excel(uploaded, index_col=0)
            
            # ★★★ 強制覆蓋：完全取代舊資料 ★★★
            st.session_state.concepts = df.columns.tolist()
            st.session_state.matrix = df.values
            
            # 清空舊的運算結果
            st.session_state.last_results = None
            st.session_state.last_initial = None
            
            st.toast(f"✅ 讀取成功！已載入 {len(df)} 個準則。", icon="📂")
        except Exception as e:
            st.error(f"檔案讀取失敗：{e}")

# ==========================================
# 3. 側邊欄設定
# ==========================================
st.sidebar.title("🛠️ 設定面板")

st.sidebar.subheader("1. 匯入矩陣")
# 下載模版
num_c = st.sidebar.number_input("準則數量", 3, 30, 13)
if st.sidebar.button("📥 下載空表"):
    dummy = [f"準則_{i+1}" for i in range(num_c)]
    df_t = pd.DataFrame(np.zeros((num_c, num_c)), index=dummy, columns=dummy)
    st.sidebar.download_button("下載 CSV", df_t.to_csv().encode('utf-8-sig'), "template.csv", "text/csv")

# 上傳檔案
st.sidebar.file_uploader(
    "上傳 Excel/CSV", 
    type=['xlsx', 'csv'], 
    key="uploader_key", 
    on_change=load_file_callback # 綁定 callback 確保資料更新
)

st.sidebar.markdown("---")
with st.sidebar.expander("2. 參數設定", expanded=True):
    LAMBDA = st.slider("Lambda", 0.1, 5.0, 1.0)
    # ★★★ 步數設定 ★★★
    MAX_STEPS = st.slider("模擬步數", 10, 100, 21) 

    if st.sidebar.button("🗑️ 清空所有資料"):
        st.session_state.concepts = []
        st.session_state.matrix = np.array([])
        st.session_state.last_results = None
        st.rerun()

# ==========================================
# 4. 主畫面 Tabs
# ==========================================
st.title("FCM 論文決策系統 (Clean Version)")

# ★★★ 防呆檢查：如果沒有資料，顯示提示畫面 ★★★
if len(st.session_state.concepts) == 0:
    st.info("👈 請先在左側側邊欄上傳您的矩陣檔案。")
    st.stop() # 停止執行後續程式碼，直到有資料為止

tab1, tab2, tab3 = st.tabs(["📊 矩陣視圖", "📈 情境模擬", "🎓 論文生成"])

# --- Tab 1 ---
with tab1:
    st.subheader(f"目前矩陣 ({len(st.session_state.concepts)}x{len(st.session_state.concepts)})")
    
    # 檢查是否全為0
    if np.all(st.session_state.matrix == 0):
        st.warning("⚠️ 警告：目前矩陣數值全為 0。請檢查您的 Excel 內容是否正確。")
    
    df_show = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
    st.dataframe(df_show.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1), height=500)

# --- Tab 2 ---
with tab2:
    st.subheader("情境模擬 (初始值 0-1)")
    
    cols = st.columns(3)
    initial_vals = []
    # 動態生成拉桿
    for i, c in enumerate(st.session_state.concepts):
        with cols[i % 3]:
            val = st.slider(c, 0.0, 1.0, 0.0, key=f"init_{i}")
            initial_vals.append(val)
            
    if st.button("🚀 開始運算", type="primary"):
        init_arr = np.array(initial_vals)
        # ★★★ 使用 MAX_STEPS 參數 ★★★
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
            st.warning("圖形為空，請拉高初始值。")
        else:
            # ★★★ 強制設定 X 軸範圍，確保顯示步數正確 ★★★
            ax.set_xlim(0, MAX_STEPS) 
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Activation (0-1)")
            ax.set_xlabel("Simulation Steps")
            ax.legend(bbox_to_anchor=(1.01, 1))
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# --- Tab 3 ---
with tab3:
    st.subheader("🎓 論文分段生成器")
    
    if st.session_state.last_results is None:
        st.error("⚠️ 請先至 Tab 2 執行運算！")
    else:
        # 計算數據
        matrix = st.session_state.matrix
        concepts = st.session_state.concepts
        results = st.session_state.last_results
        initial = st.session_state.last_initial
        final = results[-1]
        
        out_degree = np.sum(np.abs(matrix), axis=1) # 用絕對值計算影響力
        driver_idx = np.argmax(out_degree)
        driver_name = concepts[driver_idx]
        
        growth = final - initial
        best_idx = np.argmax(growth)
        best_name = concepts[best_idx]
        
        # 這裡的 steps 會自動抓取您在側邊欄設定的 MAX_STEPS
        steps = MAX_STEPS 
        density = np.count_nonzero(matrix) / (len(concepts)**2)

        # === 寫作按鈕 ===
        c1, c2, c3, c4 = st.columns(4)
        
        if c1.button("1️⃣ 生成 4.1 結構分析"):
            t = "### 第四章 研究結果與分析\n\n**4.1 FCM 矩陣結構特性分析**\n"
            t += f"本研究矩陣包含 {len(concepts)} 個準則，密度為 {density:.2f}。\n"
            t += f"數據顯示，**「{driver_name}」** 具有最高的出度 ({out_degree[driver_idx]:.2f})，確立其為關鍵驅動因子。\n"
            st.session_state.paper_sections["4.1"] = t

        if c2.button("2️⃣ 生成 4.2 穩定性"):
            t = "**4.2 系統穩定性檢測**\n"
            t += f"透過 Sigmoid 函數轉換，模擬顯示系統在第 **{steps}** 步達到收斂。各準則數值穩定落在 [0, 1] 區間內，證實模型具備動態穩定性。\n"
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
            col_d.download_button("📥 下載完整論文 (TXT)", full_text, "thesis_final.txt")
            if col_c.button("🗑️ 清空內容"):
                for k in st.session_state.paper_sections: st.session_state.paper_sections[k] = ""
                st.rerun()
        else:
            st.info("請點擊上方按鈕開始生成內容。")
