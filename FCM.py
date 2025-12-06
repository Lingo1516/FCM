import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# ==========================================
# 0. 頁面初始化 (一定要放在最第一行)
# ==========================================
st.set_page_config(page_title="FCM 論文決策系統 (最終版)", layout="wide")

# 自訂 CSS: 讓聊天室和報告更漂亮
st.markdown("""
<style>
    .chat-user { background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin: 5px; text-align: right; color: black;}
    .chat-ai { background-color: #F1F0F0; padding: 10px; border-radius: 10px; margin: 5px; text-align: left; color: black;}
    .report-card { border-left: 5px solid #2c3e50; background-color: #f8f9fa; padding: 15px; margin-bottom: 15px; border-radius: 5px; }
    .concept-title { color: #2980b9; font-weight: bold; font-size: 16px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 初始化記憶體 (Session State)
# ==========================================
if 'concepts' not in st.session_state:
    st.session_state.concepts = [
        "A1 倫理文化", "A2 高層基調", "A3 倫理風險",
        "B1 策略一致性", "B2 利害關係人", "B3 資訊透明",
        "C1 社會影響", "C2 環境責任", "C3 治理法遵"
    ]

# 初始化矩陣 (寫入論文邏輯，避免全 0)
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
    mat[3, 6] = 0.5
    mat[3, 7] = 0.6
    st.session_state.matrix = mat

# 儲存模擬結果
if 'last_results' not in st.session_state:
    st.session_state.last_results = None
    st.session_state.last_initial = None

# ★★★ 關鍵修復：初始化對話紀錄 ★★★
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    # 預設第一條歡迎訊息
    st.session_state.chat_history.append({
        "role": "ai", 
        "content": "您好，我是您的論文策略顧問。請先在「模擬運算」分頁跑出數據，然後我可以為您進行深度分析。\n\n您可以試著問我：「請解釋每一個準則的表現」或「目前的策略有什麼盲點？」"
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

# ==========================================
# 3. 側邊欄 (設定區)
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
        except Exception as e:
            st.sidebar.error(f"讀取失敗: {e}")
else:
    with st.sidebar.expander("➕ 新增準則 / 編輯"):
        new_c = st.text_input("輸入新準則名稱")
        if st.button("加入矩陣"):
            if new_c and new_c not in st.session_state.concepts:
                st.session_state.concepts.append(new_c)
                old = st.session_state.matrix
                r, c = old.shape
                new_m = np.zeros((r+1, c+1))
                new_m[:r, :c] = old
                st.session_state.matrix = new_m
                st.rerun()

LAMBDA = st.sidebar.slider("Lambda (敏感度)", 0.1, 5.0, 1.0)
MAX_STEPS = st.sidebar.slider("模擬步數", 10, 100, 30)

# ==========================================
# 4. 主畫面 (Tabs 分頁) - 修正 NameError 的關鍵
# ==========================================
st.title("FCM 論文決策系統 (AI 完整版)")

# ★★★ 先定義 Tabs，確保後面都能讀到 ★★★
tab1, tab2, tab3 = st.tabs(["📊 矩陣視圖 (Matrix)", "📈 模擬運算 (Simulate)", "🎓 AI 策略顧問 (Chatbot)"])

# --- Tab 1: 矩陣視圖 ---
with tab1:
    st.subheader("矩陣數值檢視")
    df_show = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
    st.dataframe(df_show.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1), height=400)
    st.download_button("📥 下載矩陣 CSV", df_show.to_csv().encode('utf-8'), "matrix.csv", "text/csv")

# --- Tab 2: 模擬運算 ---
with tab2:
    st.subheader("情境模擬")
    st.info("💡 請拉動下方拉桿 (設定初始策略)，再按「開始運算」。")
    
    cols = st.columns(3)
    initial_vals = []
    for i, c in enumerate(st.session_state.concepts):
        with cols[i % 3]:
            val = st.slider(c, 0.0, 1.0, 0.0, key=f"init_{i}")
            initial_vals.append(val)
    
    if st.button("🚀 開始模擬運算", type="primary"):
        init_arr = np.array(initial_vals)
        res = run_fcm(st.session_state.matrix, init_arr, LAMBDA, MAX_STEPS, 0.001)
        
        # 存入記憶體
        st.session_state.last_results = res
        st.session_state.last_initial = init_arr
        
        # 繪圖
        fig, ax = plt.subplots(figsize=(10, 5))
        active_idx = [i for i in range(len(res[0])) if res[-1, i] > 0.01 or init_arr[i] > 0]
        
        if not active_idx:
            st.warning("⚠️ 數值無變化，請嘗試增加初始投入或檢查矩陣。")
        else:
            for i in active_idx:
                ax.plot(res[:, i], label=st.session_state.concepts[i], marker='o', markersize=4)
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # 結果表
            final_v = res[-1]
            df_res = pd.DataFrame({
                "準則": st.session_state.concepts,
                "初始": init_arr,
                "最終": final_v,
                "成長": final_v - init_arr
            }).sort_values("最終", ascending=False)
            st.dataframe(df_res.style.background_gradient(cmap='Greens'))

# --- Tab 3: AI 策略顧問 (深度對話版) ---
with tab3:
    st.subheader("🤖 論文深度分析顧問")
    
    # 1. 顯示歷史對話 (解決「只能問一次」的問題)
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">👤 <b>您：</b>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-ai">🤖 <b>AI：</b>{msg["content"]}</div>', unsafe_allow_html=True)

    # 2. 輸入框
    user_input = st.text_input("請輸入您的問題...", key="chat_input")
    
    # 3. AI 處理邏輯
    if st.button("送出問題") and user_input:
        # 記錄使用者的話
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # 檢查是否有數據
        if st.session_state.last_results is None:
            response = "⚠️ 請先回到「模擬運算」分頁，執行一次模擬，我才有數據可以分析喔！"
        else:
            # 準備數據
            results = st.session_state.last_results
            final = results[-1]
            initial = st.session_state.last_initial
            concepts = st.session_state.concepts
            matrix = st.session_state.matrix
            growth = final - initial
            
            # === AI 邏輯核心 ===
            response = ""
            
            # 模式 A: 使用者想看「每一個」準則的詳細解釋
            if "每一個" in user_input or "全部" in user_input or "詳細" in user_input:
                response += "### 📊 全方位深度診斷報告\n\n"
                
                for i, c in enumerate(concepts):
                    # 分析每個準則的狀況
                    g_val = growth[i]
                    f_val = final[i]
                    init_val = initial[i]
                    
                    # 找出是誰影響了它 (In-degree)
                    influencers = []
                    col_data = matrix[:, i]
                    for idx, w in enumerate(col_data):
                        if w > 0: influencers.append(f"{concepts[idx]}(+{w})")
                    inf_str = ", ".join(influencers) if influencers else "無顯著外部驅動力"
                    
                    # 判斷學術意涵
                    status = ""
                    if init_val > 0.5:
                        status = "🔴 主動策略投入點 (Driver)"
                    elif g_val > 0.1:
                        status = "🟢 高敏感度受惠者 (Highly Sensitive)"
                    elif f_val < 0.1:
                        status = "⚪ 邊緣因子 (Inactive)"
                    else:
                        status = "🔵 一般連動因子"

                    # 組合文字
                    response += f"#### {c} {status}\n"
                    response += f"- **數據表現**：初始投入 {init_val:.1f} $\\rightarrow$ 最終收斂 {f_val:.2f} (成長 +{g_val:.2f})\n"
                    response += f"- **因果來源**：其數值變化主要受到 [{inf_str}] 的驅動。\n"
                    response += f"- **管理意涵**：{'此為本次模擬的核心策略，應持續監控其擴散效應。' if init_val > 0 else '此為被動受惠指標，無需直接投入資源，只需強化上游驅動因子即可提升。'}\n\n"
                    
                response += "\n💡 **總結**：建議論文中可將「主動策略投入點」與「高敏感度受惠者」作為因果路徑分析的重點。"

            # 模式 B: 詢問盲點或缺點
            elif "盲點" in user_input or "缺點" in user_input or "無效" in user_input:
                # 找出投入了但沒反應的 (ROI低)
                inefficient = []
                for i, val in enumerate(initial):
                    if val > 0 and growth[i] < 0.05:
                        inefficient.append(concepts[i])
                
                # 找出完全沒動的
                dead_nodes = [concepts[i] for i, f in enumerate(final) if f < 0.05]
                
                response += "### 🔍 策略盲點偵測\n\n"
                if inefficient:
                    response += f"**1. 低效率投資：** 您投入了 **{', '.join(inefficient)}**，但系統顯示其帶動效果不佳。這在學術上稱為「策略孤島 (Strategic Silo)」，暗示該準則缺乏對外的連結路徑。\n"
                else:
                    response += "**1. 投資效率：** 目前所有投入的策略皆有產生一定程度的擴散，無明顯浪費資源狀況。\n"
                    
                if dead_nodes:
                    response += f"**2. 系統死角：** **{', '.join(dead_nodes[:3])}** 等指標數值過低。若這些是重要績效，代表目前的策略組合無法觸及這些領域，這是論文中可以探討的「改進空間」。"

            # 模式 C: 一般回答
            else:
                best_idx = np.argmax(final)
                driver_idx = np.argmax(initial)
                response += f"根據模擬結果，**{concepts[best_idx]}** 是目前表現最好的指標。\n"
                response += f"這主要是由 **{concepts[driver_idx]}** 所驅動的連鎖反應。\n\n"
                response += "若您需要更詳細的個別分析，請輸入「解釋每一個準則」。"

        # 記錄 AI 的話
        st.session_state.chat_history.append({"role": "ai", "content": response})
        st.rerun() # 強制刷新畫面，顯示最新對話
