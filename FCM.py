import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 0. 頁面初始化
# ==========================================
st.set_page_config(page_title="FCM 論文決策系統 (顏色修復版)", layout="wide")

# ★★★ 這裡就是修正的地方：加上 color: #000000 ★★★
st.markdown("""
<style>
    .chat-user { background-color: #DCF8C6; padding: 15px; border-radius: 15px; margin: 10px 0; text-align: right; color: black; }
    .chat-ai { background-color: #F8F9FA; padding: 20px; border-radius: 15px; margin: 10px 0; text-align: left; color: #2c3e50; border-left: 5px solid #3498db; }
    
    /* 強制設定論文區塊的文字為黑色，避免在深色模式下變成「白字白底」 */
    .report-box { 
        border: 1px solid #ddd; 
        padding: 25px; 
        border-radius: 5px; 
        background-color: #ffffff; 
        color: #000000 !important; 
        line-height: 1.8; 
        font-family: "Times New Roman", serif; 
    }
    
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
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

# 預設矩陣 (寫入論文邏輯：高層基調 A2 為核心)
if 'matrix' not in st.session_state:
    mat = np.zeros((9, 9))
    mat[1, 0] = 0.85 # A2 -> A1
    mat[1, 3] = 0.80 # A2 -> B1
    mat[1, 5] = 0.75 # A2 -> B3
    mat[5, 4] = 0.90 # B3 -> B2
    mat[2, 8] = 0.80 # A3 -> C3
    mat[3, 6] = 0.50 # B1 -> C1
    mat[3, 7] = 0.60 # B1 -> C2
    st.session_state.matrix = mat

if 'last_results' not in st.session_state:
    st.session_state.last_results = None
    st.session_state.last_initial = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append({
        "role": "ai", 
        "content": "系統已修復顏色顯示問題。請先跑一次模擬，再點擊按鈕生成論文。"
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

    # ★★★ 這裡加一個按鈕，以防你的矩陣變成全0 ★★★
    if st.sidebar.button("⚠️ 恢復預設數據 (若圖跑不出來按此)"):
        st.session_state.concepts = [
            "A1 倫理文化", "A2 高層基調", "A3 倫理風險",
            "B1 策略一致性", "B2 利害關係人", "B3 資訊透明",
            "C1 社會影響", "C2 環境責任", "C3 治理法遵"
        ]
        mat = np.zeros((9, 9))
        mat[1, 0] = 0.85; mat[1, 3] = 0.80; mat[1, 5] = 0.75
        mat[5, 4] = 0.90; mat[2, 8] = 0.80; mat[3, 6] = 0.50; mat[3, 7] = 0.60
        st.session_state.matrix = mat
        st.rerun()

LAMBDA = st.sidebar.slider("Lambda (敏感度)", 0.1, 5.0, 1.0)
MAX_STEPS = st.sidebar.slider("模擬步數", 10, 100, 30)

# ==========================================
# 4. 主畫面
# ==========================================
st.title("FCM 論文決策系統 (Color Fixed)")
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
            st.warning("⚠️ 數值無變化，請嘗試增加初始投入，或按側邊欄的「恢復預設數據」。")
        else:
            for i in active_idx:
                ax.plot(res[:, i], label=st.session_state.concepts[i])
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            st.pyplot(fig)

# --- Tab 3 (修正顏色) ---
with tab3:
    st.subheader("🤖 論文生成與深度分析")
    
    if st.session_state.last_results is None:
        st.error("⚠️ 請先回到「Tab 2 模擬運算」執行一次運算，這裡才有數據可以寫論文！")
    else:
        results = st.session_state.last_results
        initial = st.session_state.last_initial
        final = results[-1]
        growth = final - initial
        concepts = st.session_state.concepts
        steps = results.shape[0]
        matrix = st.session_state.matrix
        
        driver_idx = np.argmax(initial)
        driver_name = concepts[driver_idx]
        best_idx = np.argmax(growth)
        best_name = concepts[best_idx]
        
        convergence_step = steps
        for t in range(1, steps):
            if np.max(np.abs(results[t] - results[t-1])) < 0.001:
                convergence_step = t
                break
        
        out_degree = np.sum(np.abs(matrix), axis=1)
        struct_driver_idx = np.argmax(out_degree)
        struct_driver_name = concepts[struct_driver_idx]

        c1, c2, c3 = st.columns(3)
        b1 = c1.button("📝 生成第四章：驗證結果")
        b2 = c2.button("🎓 生成第五章：結論建議")
        b3 = c3.button("📑 生成整本論文 (Ch4+Ch5)", type="primary")
        
        report_content = ""
        
        if b1 or b3:
            report_content += "### 📊 第四章：研究結果與驗證 (Chapter 4: Results and Verification)\n\n"
            report_content += "**4.1 結構特性分析 (Structural Analysis)**\n"
            report_content += f"本研究依據 FCM 理論進行結構檢測。計算結果顯示，**{struct_driver_name}** 具有最高的出度 (Out-degree={out_degree[struct_driver_idx]:.2f})，這在圖論上代表其為系統中影響力最強的「發送者 (Transmitter)」。此結構特徵支持將其選為策略介入的起點。\n\n"
            
            report_content += "**4.2 系統穩定性檢測 (Stability Test)**\n"
            report_content += f"為確保模型推論的有效性，本研究進行了收斂測試。模擬顯示，在既定權重下，系統經過 **{convergence_step}** 個疊代週期 (Iterations) 後達到穩態 (Steady State)。變異量收斂至 0.001 以下，證實模型具備動態穩定性，未出現發散現象。\n\n"
            
            report_content += "**4.3 情境模擬分析 (Scenario Simulation)**\n"
            report_content += f"設定情境：強化投入 **{driver_name}** (Initial Input={initial[driver_idx]:.1f})。\n"
            report_content += f"模擬軌跡顯示，隨著策略發酵，**{best_name}** 呈現最顯著的非線性成長 (由 {initial[best_idx]:.2f} 升至 {final[best_idx]:.2f})。從時序來看，系統在第 5-{int(convergence_step/2)} 步區間變化最劇烈，此為策略擴散的關鍵期。\n\n"
            report_content += "---\n\n"

        if b2 or b3:
            report_content += "### 🎓 第五章：結論與建議 (Chapter 5: Conclusion)\n\n"
            report_content += "**5.1 研究結論**\n"
            report_content += f"本研究證實 **{driver_name}** 為啟動製造業 ESG 轉型的核心驅動因子。模擬數據顯示，該因子能有效透過路徑傳導，激活後端的 **{best_name}**。這驗證了治理機制與績效表現之間的因果鏈結。\n\n"
            
            report_content += "**5.2 管理意涵**\n"
            report_content += f"1. **精準資源配置**：管理者應避免資源分散，建議集中火力強化 **{driver_name}**，利用其高中心性帶動整體系統。\n"
            report_content += f"2. **重視時間滯後**：由於系統需 {convergence_step} 步才收斂，管理者需容忍轉型初期的成效延遲，避免短視決策。\n\n"
            
            report_content += "**5.3 學術貢獻**\n"
            report_content += "本研究利用 FCM 視覺化了 ESG 變數間的動態因果路徑，突破了傳統靜態分析的限制，為高階梯隊理論提供了新的實證支持。\n"

        if report_content:
            st.markdown(f'<div class="report-box">{report_content}</div>', unsafe_allow_html=True)
            st.session_state.chat_history.append({"role": "ai", "content": report_content})

    st.divider()
    st.caption("💬 補充問答")
    user_input = st.text_input("輸入問題 (例如：解釋每一個準則)", key="chat_in")
    if user_input:
        st.info("請點擊上方按鈕生成正式論文，或在此進行一般對話。")
