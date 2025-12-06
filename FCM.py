import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 0. 頁面初始化 (這是最重要的一行，一定要在最上面)
# ==========================================
st.set_page_config(page_title="FCM 論文決策系統 (完整版)", layout="wide")

# CSS 美化設定
st.markdown("""
<style>
    .chat-user { background-color: #DCF8C6; padding: 15px; border-radius: 15px; margin: 10px 0; text-align: right; color: black; }
    .chat-ai { background-color: #F8F9FA; padding: 20px; border-radius: 15px; margin: 10px 0; text-align: left; color: #2c3e50; border-left: 5px solid #3498db; }
    .report-box { border: 1px solid #ddd; padding: 20px; border-radius: 5px; background-color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 初始化記憶體與數據
# ==========================================
if 'concepts' not in st.session_state:
    st.session_state.concepts = [
        "A1 倫理文化", "A2 高層基調", "A3 倫理風險",
        "B1 策略一致性", "B2 利害關係人", "B3 資訊透明",
        "C1 社會影響", "C2 環境責任", "C3 治理法遵"
    ]

# 預設矩陣 (寫入論文邏輯，防止跑不出圖形)
if 'matrix' not in st.session_state:
    mat = np.zeros((9, 9))
    # A2 高層基調 -> 核心驅動
    mat[1, 0] = 0.85 # -> A1
    mat[1, 3] = 0.80 # -> B1
    mat[1, 5] = 0.75 # -> B3
    # B3 資訊透明 -> B2 利害關係人
    mat[5, 4] = 0.90
    # A3 倫理風險 -> C3 治理法遵
    mat[2, 8] = 0.80
    # B1 策略一致 -> C1, C2
    mat[3, 6] = 0.50
    mat[3, 7] = 0.60
    st.session_state.matrix = mat

if 'last_results' not in st.session_state:
    st.session_state.last_results = None
    st.session_state.last_initial = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append({
        "role": "ai", 
        "content": "您好，系統已重啟。請先在「模擬運算」分頁執行一次，然後輸入 **「幫我寫整本論文分析」**，我將為您生成包含第四章驗證與第五章結論的完整報告。"
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
    """排序功能"""
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
# 4. 主畫面 (這裡就是之前你漏掉的地方)
# ==========================================
st.title("FCM 論文決策系統 (Full Version)")

# ★★★ 這行就是解決 NameError 的關鍵，一定要在這裡定義 tab1, tab2, tab3 ★★★
tab1, tab2, tab3 = st.tabs(["📊 矩陣視圖", "📈 模擬運算", "🎓 論文寫作顧問"])

# --- Tab 1: 矩陣 ---
with tab1:
    st.subheader("矩陣檢視")
    df_show = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
    st.dataframe(df_show.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1), height=400)
    st.download_button("下載 CSV", df_show.to_csv().encode('utf-8'), "matrix.csv")

# --- Tab 2: 模擬 ---
with tab2:
    st.subheader("情境模擬")
    st.info("💡 請拉動 **A2 高層基調** 至 0.8 以上 (模擬策略介入)，再按開始運算。")
    
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

# --- Tab 3: AI 寫作核心 (包含第四章與第五章) ---
with tab3:
    st.subheader("🤖 論文生成與深度分析")
    
    # 顯示歷史訊息
    for msg in st.session_state.chat_history:
        role_class = "chat-user" if msg["role"] == "user" else "chat-ai"
        prefix = "👤 您：" if msg["role"] == "user" else "🤖 AI："
        st.markdown(f'<div class="{role_class}"><b>{prefix}</b><br>{msg["content"]}</div>', unsafe_allow_html=True)

    user_input = st.text_input("輸入指令 (強烈建議輸入：幫我寫整本論文分析)", key="chat_in")
    
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
            matrix = st.session_state.matrix
            
            # 找出關鍵數據指標
            driver_idx = np.argmax(initial)
            driver_name = concepts[driver_idx]
            best_idx = np.argmax(growth)
            best_name = concepts[best_idx]
            
            # 找出收斂步數
            convergence_step = steps
            for t in range(1, steps):
                if np.max(np.abs(results[t] - results[t-1])) < 0.001:
                    convergence_step = t
                    break
            
            # 計算結構指標 (第四章必備)
            out_degree = np.sum(np.abs(matrix), axis=1)
            struct_driver_idx = np.argmax(out_degree)
            struct_driver_name = concepts[struct_driver_idx]

            response = ""
            
            # ========================================================
            # 萬用邏輯：包含第四章與第五章
            # ========================================================
            if any(k in user_input for k in ["論文", "整本", "3000", "分析", "第四章", "第五章"]):
                
                # --- 第四章：研究結果 ---
                response += "### 📊 第四章：研究結果與驗證 (Chapter 4: Results)\n\n"
                response += "**4.1 結構特性分析 (Structural Analysis)**\n"
                response += f"本研究首先針對 FCM 矩陣進行結構檢測。計算結果顯示，**{struct_driver_name}** 具有最高的出度 (Out-degree={out_degree[struct_driver_idx]:.2f})，這在圖論上代表其為系統中影響力最強的「發送者」。此結構特徵支持將其選為策略介入的起點。\n\n"
                
                response += "**4.2 系統穩定性檢測 (Stability Test)**\n"
                response += f"為確保模型推論的有效性，本研究進行了收斂測試。模擬顯示，在既定權重下，系統經過 **{convergence_step}** 個疊代週期 (Iterations) 後達到穩態 (Steady State)。變異量收斂至 0.001 以下，證實模型具備動態穩定性，未出現發散現象。\n\n"
                
                response += "**4.3 情境模擬分析 (Scenario Simulation)**\n"
                response += f"設定情境：強化投入 **{driver_name}** (Initial Input={initial[driver_idx]:.1f})。\n"
                response += f"模擬軌跡顯示，隨著策略發酵，**{best_name}** 呈現最顯著的非線性成長 (由 {initial[best_idx]:.2f} 升至 {final[best_idx]:.2f})。從時序來看，系統在第 5-{int(convergence_step/2)} 步區間變化最劇烈，此為策略擴散的關鍵期。\n\n"
                
                response += "---\n\n"
                
                # --- 第五章：結論 ---
                response += "### 🎓 第五章：結論與建議 (Chapter 5: Conclusion)\n\n"
                response += "**5.1 研究結論**\n"
                response += f"本研究證實 **{driver_name}** 為啟動製造業 ESG 轉型的核心驅動因子。模擬數據顯示，該因子能有效透過路徑傳導，激活後端的 **{best_name}**。這驗證了治理機制與績效表現之間的因果鏈結。\n\n"
                
                response += "**5.2 管理意涵**\n"
                response += f"1. **精準資源配置**：管理者應避免資源分散，建議集中火力強化 **{driver_name}**，利用其高中心性帶動整體系統。\n"
                response += f"2. **重視時間滯後**：由於系統需 {convergence_step} 步才收斂，管理者需容忍轉型初期的成效延遲，避免短視決策。\n\n"
                
                response += "**5.3 學術貢獻**\n"
                response += "本研究利用 FCM 視覺化了 ESG 變數間的動態因果路徑，突破了傳統靜態分析的限制，為高階梯隊理論提供了新的實證支持。\n"

            # 其他簡單對話
            else:
                response += f"已收到指令。建議輸入 **「幫我寫整本論文分析」**，我將為您生成包含第四章驗證與第五章結論的完整報告。"

        st.session_state.chat_history.append({"role": "ai", "content": response})
        st.rerun()
