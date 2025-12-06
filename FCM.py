import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 0. 初始化狀態與樣式
# ==========================================
st.set_page_config(page_title="FCM 智慧決策系統", layout="wide")

# 自訂 CSS 讓介面更像專業軟體
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #f0f2f6; border-radius: 4px; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #e6ffe6; border-bottom: 2px solid green; }
    .big-font { font-size:20px !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

if 'concepts' not in st.session_state:
    st.session_state.concepts = [
        "A1 倫理文化", "A2 高層基調", "A3 倫理風險",
        "B1 策略一致性", "B2 利害關係人", "B3 資訊透明",
        "C1 社會影響", "C2 環境責任", "C3 治理法遵"
    ]

if 'matrix' not in st.session_state:
    st.session_state.matrix = np.zeros((9, 9))
    # 預設邏輯 (高層基調 A2 是核心)
    weights = st.session_state.matrix
    weights[1, 0] = 0.85 
    weights[1, 3] = 0.8
    weights[1, 5] = 0.7

# 儲存最後一次的模擬結果供 AI 分析用
if 'last_results' not in st.session_state:
    st.session_state.last_results = None
if 'last_initial' not in st.session_state:
    st.session_state.last_initial = None

# ==========================================
# 1. 功能函數 (排序與運算)
# ==========================================
def sort_matrix_and_concepts():
    """核心功能：依照名稱 (A1, A2...) 自動排序，並確保矩陣數值跟著搬家"""
    # 1. 先把目前的矩陣變成 DataFrame (有名字的表)
    df = pd.DataFrame(
        st.session_state.matrix, 
        index=st.session_state.concepts, 
        columns=st.session_state.concepts
    )
    
    # 2. 進行排序 (Sort) - 橫向縱向同時排
    df_sorted = df.sort_index(axis=0).sort_index(axis=1)
    
    # 3. 存回 Session State
    st.session_state.concepts = df_sorted.index.tolist()
    st.session_state.matrix = df_sorted.values
    st.success("✅ 矩陣已重新排序！(例如 A4 已自動插入 A3 後方)")

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
# 2. 側邊欄：控制面板
# ==========================================
st.sidebar.title("🛠️ 控制面板")

# --- 矩陣來源切換 ---
data_source = st.sidebar.radio("矩陣模式", ["隨機/編輯模式", "上傳 Excel"])

if data_source == "上傳 Excel":
    uploaded_file = st.sidebar.file_uploader("上傳 .xlsx/.csv", type=['xlsx', 'csv'])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, index_col=0)
            else:
                df = pd.read_excel(uploaded_file, index_col=0)
            st.session_state.concepts = df.columns.tolist()
            st.session_state.matrix = df.values
            st.sidebar.success("檔案讀取成功")
        except:
            st.sidebar.error("格式錯誤")

else:
    # --- 編輯功能區 ---
    col_add, col_sort = st.sidebar.columns(2)
    
    # 新增準則
    with st.sidebar.expander("➕ 新增/管理準則", expanded=True):
        new_concept = st.text_input("輸入名稱 (如: A4 人才培訓)")
        if st.button("加入矩陣"):
            if new_concept and new_concept not in st.session_state.concepts:
                st.session_state.concepts.append(new_concept)
                # 擴充矩陣
                old = st.session_state.matrix
                r, c = old.shape
                new_m = np.zeros((r+1, c+1))
                new_m[:r, :c] = old
                st.session_state.matrix = new_m
                st.success(f"已新增 {new_concept} (在最後面)")
                st.rerun()
        
        # 排序按鈕 (這就是你要的功能！)
        if st.button("🔄 自動排序 (Sort A-Z)"):
            sort_matrix_and_concepts()
            st.rerun()

    # 隨機生成
    if st.sidebar.button("🎲 隨機生成權重"):
        n = len(st.session_state.concepts)
        rand = np.random.uniform(-0.5, 0.8, (n, n))
        np.fill_diagonal(rand, 0)
        rand[np.abs(rand) < 0.2] = 0
        st.session_state.matrix = rand
        st.sidebar.success("已生成隨機權重")

st.sidebar.markdown("---")
# 參數
LAMBDA = st.sidebar.slider("Lambda (敏感度)", 0.1, 5.0, 1.0)
MAX_STEPS = st.sidebar.slider("模擬步數", 10, 100, 30)

# ==========================================
# 3. 主畫面：分頁設計
# ==========================================
st.title("FCM 智慧決策系統")

tab1, tab2, tab3 = st.tabs(["📊 矩陣視圖 (Matrix)", "📈 模擬運算 (Simulation)", "🤖 AI 策略顧問 (Analyst)"])

# --- Tab 1: 矩陣視圖 ---
with tab1:
    st.subheader("目前系統架構矩陣")
    st.caption("您可以直接在此確認排序是否正確，以及數值分佈。")
    
    df_display = pd.DataFrame(
        st.session_state.matrix,
        index=st.session_state.concepts,
        columns=st.session_state.concepts
    )
    # 用熱力圖顏色顯示 (藍正紅負)
    st.dataframe(df_display.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1), height=500)
    
    # 下載功能
    csv = df_display.to_csv().encode('utf-8')
    st.download_button("📥 下載此矩陣 (CSV)", csv, "fcm_matrix.csv", "text/csv")

# --- Tab 2: 模擬運算 ---
with tab2:
    st.subheader("情境模擬設定")
    
    # 初始值拉桿
    st.info("請設定初始策略投入 (0=無作為, 1=全力投入)")
    cols = st.columns(3)
    initial_vals = []
    for i, c in enumerate(st.session_state.concepts):
        with cols[i % 3]:
            val = st.slider(c, 0.0, 1.0, 0.0, key=f"init_{i}")
            initial_vals.append(val)
    
    if st.button("🚀 開始模擬運算", type="primary"):
        init_arr = np.array(initial_vals)
        res = run_fcm(st.session_state.matrix, init_arr, LAMBDA, MAX_STEPS, 0.001)
        
        # 存起來給 AI 分析用
        st.session_state.last_results = res
        st.session_state.last_initial = init_arr
        
        # 繪圖
        fig, ax = plt.subplots(figsize=(10, 5))
        # 偵測有變動的線才畫
        active_idx = [i for i in range(len(res[0])) if res[-1, i] > 0.01 or init_arr[i] > 0]
        
        if not active_idx:
            st.warning("⚠️ 數值無變化，請嘗試增加初始投入或檢查矩陣連結。")
        else:
            for i in active_idx:
                ax.plot(res[:, i], label=st.session_state.concepts[i], marker='o', markersize=3)
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_title("動態收斂過程")
            st.pyplot(fig)
            
            # 數據表
            final_s = res[-1]
            df_res = pd.DataFrame({
                "準則": st.session_state.concepts,
                "初始": init_arr,
                "最終": final_s,
                "成長": final_s - init_arr
            }).sort_values("最終", ascending=False)
            st.dataframe(df_res.style.background_gradient(cmap='Greens'))

# --- Tab 3: AI 策略顧問 (你要求的問答功能) ---
with tab3:
    st.subheader("🤖 AI 策略顧問對話視窗")
    
    if st.session_state.last_results is None:
        st.warning("請先在「模擬運算」分頁執行一次模擬，我才能分析數據。")
    else:
        # 準備數據
        final_state = st.session_state.last_results[-1]
        initial_state = st.session_state.last_initial
        concepts = st.session_state.concepts
        matrix = st.session_state.matrix
        
        # 1. 自動診斷報告 (Auto-generated Report)
        st.markdown("### 📊 自動診斷報告")
        
        # 找出最高績效
        best_idx = np.argmax(final_state)
        worst_idx = np.argmin(final_state)
        
        # 找出無效投資 (投入了但成長很少)
        growth = final_state - initial_state
        # 避免除以零
        roi = np.divide(growth, initial_state, out=np.zeros_like(growth), where=initial_state!=0)
        inefficient_idx = np.argmin(roi) if np.any(initial_state > 0) else -1
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**表現最佳指標：** {concepts[best_idx]} (數值: {final_state[best_idx]:.2f})")
        with col2:
            if inefficient_idx != -1 and roi[inefficient_idx] < 0.1:
                st.error(f"**效率最低策略：** {concepts[inefficient_idx]} (投入高但成長低，建議檢討)")
            else:
                st.info("所有投入策略皆有產生一定成效。")

        st.markdown("---")

        # 2. 互動問答區
        st.markdown("### 💬 請問您的問題")
        st.caption("您可以詢問關於策略調整、圖形解釋或異常分析的問題。")
        
        user_question = st.text_input("輸入問題 (例如：哪個策略無效？如何改善 C2？)", "")
        
        if user_question:
            st.markdown("#### 🤖 AI 回答：")
            
            # === 這裡模擬 AI 的邏輯判斷 (Rule-Based AI) ===
            response = ""
            
            if "無效" in user_question or "錯" in user_question or "失敗" in user_question:
                low_growth_indices = [i for i, g in enumerate(growth) if g < 0.05 and initial_state[i] > 0]
                if low_growth_indices:
                    names = [concepts[i] for i in low_growth_indices]
                    response = f"根據模擬數據，以下策略似乎陷入瓶頸：**{', '.join(names)}**。\n\n原因可能是：\n1. 這些準則在矩陣中缺乏強大的正向連結。\n2. 受到其他負面因子的抑制 (負權重)。\n\n建議：檢查矩陣中這些列 (Row) 的數值是否過低。"
                else:
                    response = "目前的模擬顯示策略皆有正面產出，沒有明顯失敗的策略。若覺得成長不夠快，建議提高 Lambda 值或增強矩陣權重。"
            
            elif "解釋" in user_question or "圖" in user_question:
                response = f"這張圖表顯示了系統從初始狀態到收斂的過程。\n\n- **X軸** 代表時間步數 (Steps)。\n- **Y軸** 代表該概念被激活的程度 (0~1)。\n\n目前的趨勢顯示，**{concepts[best_idx]}** 是系統中的領導者，它的上升帶動了整體效能。若線條呈現平緩，代表系統已達穩定狀態。"
                
            elif "如何" in user_question and "改善" in user_question:
                # 簡單分析矩陣，找出誰能影響目標
                target = None
                for c in concepts:
                    if c in user_question: # 嘗試抓使用者問的概念
                        target = c
                        break
                
                if target:
                    t_idx = concepts.index(target)
                    # 找誰影響它最大 (Column search)
                    influencers = matrix[:, t_idx]
                    top_inf_idx = np.argmax(influencers)
                    
                    if influencers[top_inf_idx] > 0:
                        response = f"若要改善 **{target}**，最有效的方法不是直接投入它，而是強化 **{concepts[top_inf_idx]}**。\n\n數據顯示 {concepts[top_inf_idx]} 對 {target} 有最強的正向影響力 (權重 {influencers[top_inf_idx]:.2f})。"
                    else:
                        response = f"**{target}** 目前似乎缺乏強大的外部驅動力 (沒有其他概念顯著正向影響它)。建議修改矩陣，增加對它的影響權重。"
                else:
                    response = "若要改善特定指標，請在問題中明確指出指標名稱 (例如：如何改善 C2 環境責任？)。一般而言，強化『高層基調 (A2)』通常能帶動整體 ESG 表現。"
            
            else:
                response = "這是一個很好的問題。根據 FCM 理論，您可以嘗試：\n1. 調整初始投入值，觀察「敏感度分析」。\n2. 檢查矩陣中的負數，看是否有互相抵銷的狀況。\n\n(若需更深入的語意分析，未來可串接 OpenAI API)"
                
            st.info(response)
