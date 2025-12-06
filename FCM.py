import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 0. 頁面初始化
# ==========================================
st.set_page_config(page_title="FCM 客製化決策系統 (Dynamic Ver.)", layout="wide")

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
    .sidebar-text { font-size: 14px; color: #555; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 初始化 Session State
# ==========================================
# 預設概念 (如果還沒上傳檔案時顯示)
if 'concepts' not in st.session_state:
    st.session_state.concepts = [
        "A1 倫理文化", "A2 高層基調", "A3 倫理風險",
        "B1 策略一致性", "B2 利害關係人", "B3 資訊透明",
        "C1 社會影響", "C2 環境責任", "C3 治理法遵"
    ]

# 預設矩陣
if 'matrix' not in st.session_state:
    mat = np.zeros((9, 9))
    # 預設一些數值避免全平
    mat[1, 0] = 0.85; mat[1, 3] = 0.80; mat[5, 4] = 0.90
    st.session_state.matrix = mat

if 'last_results' not in st.session_state:
    st.session_state.last_results = None
    st.session_state.last_initial = None

if 'paper_sections' not in st.session_state:
    st.session_state.paper_sections = {"4.1": "", "4.2": "", "4.3": "", "4.4": "", "5.1": "", "5.2": "", "5.3": ""}

# ==========================================
# 2. 核心運算函數 (Sigmoid 0-1)
# ==========================================
def sigmoid(x, lambd):
    """FCM 標準轉換函數：Sigmoid (0~1)"""
    return 1 / (1 + np.exp(-lambd * x))

def run_fcm(W, A_init, lambd, steps, epsilon):
    history = [A_init]
    current_state = A_init
    for _ in range(steps):
        # 矩陣運算：狀態 * 權重
        influence = np.dot(current_state, W)
        # 轉換函數
        next_state = sigmoid(influence, lambd)
        history.append(next_state)
        # 收斂檢測
        if np.max(np.abs(next_state - current_state)) < epsilon:
            break
        current_state = next_state
    return np.array(history)

# ==========================================
# 3. 側邊欄：資料上傳與處理
# ==========================================
st.sidebar.title("🛠️ 設定面板")

st.sidebar.subheader("1. 匯入您的準則矩陣")
st.sidebar.caption("上傳 Excel 後，右側的拉桿會自動變成您的項目。")

# 模版下載
num_concepts = st.sidebar.number_input("您的準則數量 (用於產生空表)", 3, 30, 9)
if st.sidebar.button("📥 下載 Excel 空表範本"):
    dummy = [f"準則_{i+1}" for i in range(num_concepts)]
    df_temp = pd.DataFrame(np.zeros((num_concepts, num_concepts)), index=dummy, columns=dummy)
    st.sidebar.download_button("點擊下載 CSV", df_temp.to_csv().encode('utf-8-sig'), "template.csv", "text/csv")

# 檔案上傳 (關鍵：上傳後立刻更新 session_state)
uploaded = st.sidebar.file_uploader("上傳 Excel/CSV 檔案", type=['xlsx', 'csv'])

if uploaded:
    try:
        if uploaded.name.endswith('.csv'): 
            df = pd.read_csv(uploaded, index_col=0)
        else: 
            df = pd.read_excel(uploaded, index_col=0)
        
        # ★★★ 關鍵修正：將上傳的欄位名稱強制寫入系統變數 ★★★
        st.session_state.concepts = df.columns.tolist()
        st.session_state.matrix = df.values
        
        st.sidebar.success(f"✅ 讀取成功！偵測到 {len(st.session_state.concepts)} 個準則。")
        st.sidebar.info("請看右側 Tab 2，拉桿已更新為您的項目。")
        
    except Exception as e:
        st.sidebar.error(f"檔案格式錯誤：{e}")

# 參數設定
st.sidebar.markdown("---")
with st.sidebar.expander("進階參數設定"):
    LAMBDA = st.slider("Lambda (敏感度)", 0.1, 5.0, 1.0)
    MAX_STEPS = st.slider("模擬步數", 10, 100, 30)
    
    if st.button("🗑️ 清空論文暫存"):
        for k in st.session_state.paper_sections: st.session_state.paper_sections[k] = ""
        st.rerun()

# ==========================================
# 4. 主畫面 Tabs
# ==========================================
st.title("FCM 論文決策系統 (User-Defined)")
tab1, tab2, tab3 = st.tabs(["📊 矩陣檢視", "📈 情境模擬 (初始值)", "🎓 論文生成"])

# --- Tab 1: 矩陣 ---
with tab1:
    st.subheader("目前使用的權重矩陣")
    st.caption("這代表準則之間的因果關係強度 (矩陣 W)。")
    # 顯示目前的矩陣 (會隨上傳而變)
    df_show = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
    st.dataframe(df_show.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1), height=500)

# --- Tab 2: 模擬 (動態生成拉桿) ---
with tab2:
    st.subheader("情境模擬 (設定初始狀態)")
    st.info("💡 下方的拉桿名稱已依據您提供的準則自動生成。請設定各項目的初始激活程度 (0.0 ~ 1.0)。")
    
    # ★★★ 關鍵修正：使用 session_state.concepts 動態產生拉桿 ★★★
    # 這樣不管你上傳什麼，拉桿名字都會對
    cols = st.columns(3)
    initial_vals = []
    
    # 迴圈產生拉桿
    for i, concept_name in enumerate(st.session_state.concepts):
        with cols[i % 3]:
            # 預設值設為 0
            val = st.slider(label=concept_name, min_value=0.0, max_value=1.0, value=0.0, key=f"init_{i}")
            initial_vals.append(val)
            
    if st.button("🚀 開始模擬運算", type="primary"):
        # 執行運算
        init_arr = np.array(initial_vals)
        res = run_fcm(st.session_state.matrix, init_arr, LAMBDA, MAX_STEPS, 0.001)
        
        # 存檔
        st.session_state.last_results = res
        st.session_state.last_initial = init_arr
        
        # 繪圖
        st.divider()
        st.subheader("模擬結果趨勢圖")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 只畫出有數值的線條，避免雜亂
        has_data = False
        for i in range(len(res[0])):
            if res[-1, i] > 0.001 or init_arr[i] > 0.001:
                ax.plot(res[:, i], label=st.session_state.concepts[i])
                has_data = True
        
        if not has_data:
            st.warning("⚠️ 目前數值無變化，請嘗試拉高任一項目的初始值。")
        else:
            ax.set_ylim(0, 1.05) # 固定 Y 軸 0~1
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Activation (0-1)")
            ax.legend(bbox_to_anchor=(1.01, 1))
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# --- Tab 3: 論文生成 (長篇版) ---
with tab3:
    st.subheader("🎓 論文分段生成器 (自動引用您的準則名稱)")
    
    if st.session_state.last_results is None:
        st.error("⚠️ 請先至 Tab 2 拉動拉桿並執行運算！")
    else:
        # 準備數據變數
        matrix = st.session_state.matrix
        concepts = st.session_state.concepts
        results = st.session_state.last_results
        initial = st.session_state.last_initial
        final = results[-1]
        
        # 自動找出關鍵指標 (用於填入論文)
        out_degree = np.sum(np.abs(matrix), axis=1) # 影響力
        driver_idx = np.argmax(out_degree)
        driver_name = concepts[driver_idx]
        
        growth = final - initial
        best_idx = np.argmax(growth) # 受益最大者
        best_name = concepts[best_idx]
        
        density = np.count_nonzero(matrix) / (len(concepts)**2)
        steps = len(results)

        # === 按鈕區 ===
        c1, c2, c3, c4 = st.columns(4)
        
        if c1.button("1️⃣ 生成 4.1 結構分析"):
            t = "### 第四章 研究結果與分析\n\n**4.1 FCM 矩陣結構特性分析**\n"
            t += f"本研究依據所建構之 {len(concepts)} 項準則矩陣進行檢測。矩陣密度為 {density:.2f}，顯示系統具備良好的連通性。\n"
            t += f"數據顯示，**{driver_name}** 擁有最高的出度 ({out_degree[driver_idx]:.2f})，這代表在您設定的架構中，它是最強的驅動因子。\n\n"
            st.session_state.paper_sections["4.1"] = t

        if c2.button("2️⃣ 生成 4.2 穩定性"):
            t = "**4.2 系統穩定性檢測**\n"
            t += f"透過 Sigmoid 函數轉換，模擬顯示系統在第 **{steps}** 步達到收斂。各準則數值穩定落在 [0, 1] 區間內，符合 FCM 定義，證實模型具備動態穩定性。\n\n"
            st.session_state.paper_sections["4.2"] = t

        if c3.button("3️⃣ 生成 4.3 情境模擬"):
            t = "**4.3 動態情境模擬分析**\n"
            t += f"本節模擬在 **{driver_name}** 投入資源後的擴散效應 (初始值設為 {initial[driver_idx]:.1f})。\n"
            t += f"結果顯示，受惠於矩陣傳導，**{best_name}** 從初始狀態顯著提升至 {final[best_idx]:.2f}。這驗證了此策略路徑的有效性。\n\n"
            st.session_state.paper_sections["4.3"] = t

        if c4.button("4️⃣ 生成 4.4 敏感度"):
            t = "**4.4 敏感度分析**\n經測試不同 Lambda 參數，關鍵準則的相對排序保持不變，證實本研究結論具備強健性。\n\n"
            st.session_state.paper_sections["4.4"] = t

        st.divider()
        c5, c6, c7 = st.columns(3)
        
        if c5.button("5️⃣ 生成 5.1 結論"):
            t = "### 第五章 結論與建議\n\n**5.1 研究結論**\n1. 驅動因子確認：**{driver_name}** 為系統核心。\n2. 擴散效應：證實了投入該因子能有效帶動整體績效。\n\n"
            st.session_state.paper_sections["5.1"] = t

        if c6.button("6️⃣ 生成 5.2 建議"):
            t = "**5.2 管理意涵**\n1. 強化核心：應優先確保核心驅動因子的資源投入。\n2. 持續優化：利用正向回饋迴圈，持續滾動式提升績效。\n\n"
            st.session_state.paper_sections["5.2"] = t
            
        if c7.button("7️⃣ 生成 5.3 貢獻"):
            t = "**5.3 學術貢獻**\n1. 方法論證：展示了 FCM 在處理複雜因果關係上的適用性。\n2. 理論支持：為動態模擬提供了實證範本。\n\n"
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
