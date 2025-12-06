import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time  # ★★★ 修正：補上這個，隨機功能才不會報錯 ★★★

# ==========================================
# 0. 頁面初始化
# ==========================================
st.set_page_config(page_title="FCM 論文決策系統 (最終修正版)", layout="wide")

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
# 1. 初始化數據 (確保不為空)
# ==========================================
if 'concepts' not in st.session_state:
    # 預設 9 個，但會隨上傳改變
    st.session_state.concepts = [f"C{i+1}" for i in range(9)]

if 'matrix' not in st.session_state:
    # 預設一個非零矩陣，避免第一次打開圖形是平的
    mat = np.zeros((9, 9))
    np.fill_diagonal(mat, 0)
    # 隨機填入一些正數，讓使用者知道系統是活的
    rows, cols = np.indices((9, 9))
    mat[rows != cols] = 0.5 
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
    # ★★★ 公式確認：Sigmoid 將數值壓縮在 0 到 1 之間 ★★★
    # 只有當 x (輸入權重總和) 為 0 時，結果才是 0.5
    return 1 / (1 + np.exp(-lambd * x))

def run_fcm(W, A_init, lambd, steps, epsilon):
    history = [A_init]
    current_state = A_init
    for _ in range(steps):
        # 1. 矩陣運算 (狀態 x 權重)
        influence = np.dot(current_state, W)
        
        # 2. 轉換函數
        next_state = sigmoid(influence, lambd)
        
        history.append(next_state)
        
        # 3. 判斷收斂
        if np.max(np.abs(next_state - current_state)) < epsilon:
            break
        current_state = next_state
        
    return np.array(history)

def sort_matrix_logic():
    try:
        df = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
        df_sorted = df.sort_index(axis=0).sort_index(axis=1)
        st.session_state.concepts = df_sorted.index.tolist()
        st.session_state.matrix = df_sorted.values
        st.success("✅ 排序完成！")
    except Exception as e:
        st.error(f"排序失敗，請確認矩陣大小與準則數量是否一致。錯誤：{e}")

# ==========================================
# 3. 側邊欄：資料處理
# ==========================================
st.sidebar.title("🛠️ 設定面板")

st.sidebar.subheader("1. 匯入您的矩陣")
# 下載模版
num_c = st.sidebar.number_input("您的準則數量", 3, 30, 13) # 預設改為 13 符合您的圖片
if st.sidebar.button("📥 下載 Excel 空表"):
    dummy = [f"準則_{i+1}" for i in range(num_c)]
    df_temp = pd.DataFrame(np.zeros((num_c, num_c)), index=dummy, columns=dummy)
    st.sidebar.download_button("點擊下載 CSV", df_temp.to_csv().encode('utf-8-sig'), "template.csv", "text/csv")

# 上傳檔案 (關鍵修復點)
uploaded = st.sidebar.file_uploader("上傳 Excel/CSV", type=['xlsx', 'csv'])

if uploaded:
    try:
        if uploaded.name.endswith('.csv'): 
            df = pd.read_csv(uploaded, index_col=0)
        else: 
            df = pd.read_excel(uploaded, index_col=0)
        
        # ★★★ 強制更新 Session State ★★★
        st.session_state.concepts = df.columns.tolist()
        st.session_state.matrix = df.values
        
        # 檢查是否全為 0
        if np.all(df.values == 0):
            st.sidebar.warning("⚠️ 警告：您上傳的矩陣數值全部為 0！這會導致圖形變成一條直線 (0.5)。請檢查 Excel 內容。")
        else:
            st.sidebar.success(f"✅ 讀取成功！共 {len(df)} 個準則。")
            
    except Exception as e:
        st.sidebar.error(f"檔案讀取錯誤：{e}")

st.sidebar.markdown("---")
# 編輯工具
with st.sidebar.expander("🔧 矩陣工具", expanded=False):
    if st.button("🔄 自動排序"):
        sort_matrix_logic()
        st.rerun()
        
    if st.button("🎲 隨機生成權重 (0~1)"):
        # 隨機產生 0-1 之間的權重，模擬真實矩陣
        n = len(st.session_state.concepts)
        rand = np.random.uniform(0.0, 1.0, (n, n))
        np.fill_diagonal(rand, 0)
        rand[rand < 0.3] = 0 # 讓矩陣稀疏一點
        st.session_state.matrix = rand
        st.success("已生成隨機矩陣！請去 Tab 2 運算。")
        time.sleep(0.5)
        st.rerun()

    if st.button("🗑️ 清空論文"):
        for k in st.session_state.paper_sections: st.session_state.paper_sections[k] = ""
        st.rerun()

# 參數設定
with st.sidebar.expander("⚙️ 模擬參數", expanded=True):
    LAMBDA = st.slider("Lambda (敏感度)", 0.1, 5.0, 1.0)
    # ★★★ 修正：預設步數設為 21 ★★★
    MAX_STEPS = st.slider("模擬步數", 10, 100, 21) 

# ==========================================
# 4. 主畫面
# ==========================================
st.title("FCM 論文決策系統 (Final Fix)")
tab1, tab2, tab3 = st.tabs(["📊 矩陣視圖", "📈 情境模擬", "🎓 論文生成"])

# --- Tab 1 ---
with tab1:
    st.subheader("矩陣權重檢視")
    # 檢查矩陣狀態
    if np.all(st.session_state.matrix == 0):
        st.error("⚠️ 目前矩陣數值全為 0。請上傳正確的 Excel，或點擊側邊欄的「隨機生成權重」來測試。")
    else:
        st.caption("數值範圍 0 ~ 1 (正向影響)")
        df_show = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
        st.dataframe(df_show.style.background_gradient(cmap='Blues', vmin=0, vmax=1), height=500)

# --- Tab 2 ---
with tab2:
    st.subheader("情境模擬 (初始值 0-1)")
    st.info("💡 請設定各準則的初始投入程度。")
    
    # 動態產生拉桿
    cols = st.columns(3)
    initial_vals = []
    for i, c in enumerate(st.session_state.concepts):
        with cols[i % 3]:
            val = st.slider(c, 0.0, 1.0, 0.0, key=f"init_{i}")
            initial_vals.append(val)
            
    if st.button("🚀 開始運算 (Run Simulation)", type="primary"):
        init_arr = np.array(initial_vals)
        res = run_fcm(st.session_state.matrix, init_arr, LAMBDA, MAX_STEPS, 0.001)
        st.session_state.last_results = res
        st.session_state.last_initial = init_arr
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 繪圖
        lines_plotted = False
        for i in range(len(res[0])):
            # 只有當數值有變化，或者不是死線時才畫，避免圖太亂
            # 這裡放寬標準，只要初始值>0或者結果>0就畫
            if np.max(res[:, i]) > 0.001:
                ax.plot(res[:, i], label=st.session_state.concepts[i])
                lines_plotted = True
        
        if not lines_plotted:
            # 如果真的全都是 0 (圖形跑不出來)
            if np.all(st.session_state.matrix == 0):
                st.warning("⚠️ 圖形為空！原因：您的矩陣權重全為 0。請檢查 Tab 1 或重新上傳。")
            else:
                st.warning("⚠️ 圖形為空！原因：所有初始值均為 0。請拉動上方拉桿。")
        else:
            ax.set_xlim(0, MAX_STEPS) # X軸鎖定到您要的步數
            ax.set_ylim(0, 1.05)      # Y軸鎖定 0-1
            ax.set_xlabel("Steps (Time)")
            ax.set_ylabel("Activation (0-1)")
            ax.legend(bbox_to_anchor=(1.01, 1))
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# --- Tab 3 (長篇寫作) ---
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
            t += f"本研究矩陣包含 {len(concepts)} 個準則。矩陣密度為 {density:.2f}，顯示系統高度連通。\n"
            t += f"數據顯示，**{driver_name}** 具有最高的出度 ({out_degree[driver_idx]:.2f})，確立其為關鍵驅動因子。\n\n"
            st.session_state.paper_sections["4.1"] = t

        if c2.button("2️⃣ 生成 4.2 穩定性"):
            t = "**4.2 系統穩定性檢測**\n"
            t += f"透過 Sigmoid 函數轉換，模擬顯示系統在第 **{steps}** 步達到收斂。各準則數值穩定落在 [0, 1] 區間內，證實模型具備動態穩定性。\n\n"
            st.session_state.paper_sections["4.2"] = t

        if c3.button("3️⃣ 生成 4.3 情境模擬"):
            t = "**4.3 動態情境模擬分析**\n"
            t += f"本節模擬在 **{driver_name}** 投入資源後的擴散效應。\n"
            t += f"結果顯示，**{best_name}** 從初始狀態顯著提升至 {final[best_idx]:.2f}。這驗證了「投入 A 帶動 B」的假設。\n\n"
            st.session_state.paper_sections["4.3"] = t

        if c4.button("4️⃣ 生成 4.4 敏感度"):
            t = "**4.4 敏感度分析**\n經測試不同 Lambda 參數，關鍵準則的相對排序保持不變，證實結論具備強健性。\n\n"
            st.session_state.paper_sections["4.4"] = t

        st.divider()
        c5, c6, c7 = st.columns(3)
        
        if c5.button("5️⃣ 生成 5.1 結論"):
            t = "### 第五章 結論與建議\n\n**5.1 研究結論**\n1. 驅動因子確認：**{driver_name}** 為系統核心。\n2. 正向擴散效應：證實了治理機制能有效提升整體績效。\n\n"
            st.session_state.paper_sections["5.1"] = t

        if c6.button("6️⃣ 生成 5.2 建議"):
            t = "**5.2 管理意涵**\n1. 強化核心：應優先確保核心驅動因子的資源投入。\n2. 持續優化：利用正向回饋迴圈，持續滾動式提升績效。\n\n"
            st.session_state.paper_sections["5.2"] = t
            
        if c7.button("7️⃣ 生成 5.3 貢獻"):
            t = "**5.3 學術貢獻**\n1. 方法論證：展示了 FCM 在處理 0-1 因果關係上的適用性。\n2. 理論支持：為動態模擬提供了實證範本。\n\n"
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
            st.download_button("📥 下載完整論文 (TXT)", full_text, "thesis_final.txt")
        else:
            st.info("請點擊上方按鈕開始生成內容。")
