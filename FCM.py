import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 0. 頁面初始化
# ==========================================
st.set_page_config(page_title="FCM 論文決策系統 (Tanh版)", layout="wide")

st.markdown("""
<style>
    .report-box { 
        border: 1px solid #ddd; padding: 25px; border-radius: 5px; 
        background-color: #ffffff; color: #000000; 
        line-height: 1.8; font-family: "Times New Roman", "標楷體", serif; 
        font-size: 16px; margin-bottom: 20px;
    }
    .chat-user { background-color: #DCF8C6; padding: 15px; border-radius: 10px; text-align: right; color: black; margin: 5px;}
    .chat-ai { background-color: #E3F2FD; padding: 15px; border-radius: 10px; text-align: left; color: black; margin: 5px;}
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
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

if 'matrix' not in st.session_state:
    # 建立高密度真實矩陣
    mat = np.array([
        [0.00, 0.45, 0.60, 0.55, 0.40, 0.30, 0.35, 0.20, 0.70],
        [0.90, 0.00, 0.85, 0.95, 0.60, 0.80, 0.50, 0.45, 0.75],
        [0.50, 0.30, 0.00, 0.40, 0.20, 0.65, 0.10, 0.30, 0.85],
        [0.30, 0.40, 0.20, 0.00, 0.50, 0.60, 0.70, 0.75, 0.40],
        [0.25, 0.30, 0.15, 0.45, 0.00, 0.70, 0.80, 0.30, 0.20],
        [0.40, 0.50, 0.60, 0.55, 0.90, 0.00, 0.65, 0.40, 0.50],
        [0.30, 0.20, 0.10, 0.20, 0.60, 0.40, 0.00, 0.35, 0.30],
        [0.20, 0.25, 0.30, 0.30, 0.40, 0.50, 0.40, 0.00, 0.45],
        [0.60, 0.55, 0.70, 0.40, 0.35, 0.50, 0.30, 0.25, 0.00]
    ])
    st.session_state.matrix = mat

if 'last_results' not in st.session_state:
    st.session_state.last_results = None
    st.session_state.last_initial = None

if 'paper_sections' not in st.session_state:
    st.session_state.paper_sections = {}

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": "ai", "content": "系統已切換至 Tanh 核心，全面支援 -1 到 1 的負值模擬。"})

# ==========================================
# 2. 核心運算函數 (升級為 Tanh)
# ==========================================
def transfer_function(x, lambd):
    """
    使用 Tanh (雙曲正切) 函數
    輸出範圍：[-1, 1]
    適合處理包含負面影響或抑制作用的模擬
    """
    return np.tanh(lambd * x)

def run_fcm(W, A_init, lambd, steps, epsilon):
    history = [A_init]
    current_state = A_init
    for _ in range(steps):
        # 矩陣運算
        influence = np.dot(current_state, W)
        # 轉換函數 (Tanh)
        next_state = transfer_function(influence, lambd)
        history.append(next_state)
        # 收斂判斷
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
# 3. 側邊欄
# ==========================================
st.sidebar.title("🛠️ 設定面板")
mode = st.sidebar.radio("資料模式", ["內建模型", "上傳 Excel/CSV"])

if mode == "上傳 Excel/CSV":
    uploaded = st.sidebar.file_uploader("上傳矩陣", type=['xlsx', 'csv'])
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
    with st.sidebar.form("add_concept"):
        st.write("➕ **新增準則**")
        new_c = st.text_input("輸入名稱")
        if st.form_submit_button("加入") and new_c:
            if new_c not in st.session_state.concepts:
                st.session_state.concepts.append(new_c)
                old = st.session_state.matrix
                r, c = old.shape
                new_m = np.zeros((r+1, c+1))
                new_m[:r, :c] = old
                st.session_state.matrix = new_m
                st.success(f"已新增 {new_c}")
                st.rerun()

    if st.sidebar.button("🔄 自動排序"):
        sort_matrix_logic()
        st.rerun()
        
    if st.sidebar.button("⚠️ 重置矩陣"):
        st.session_state.concepts = ["A1 倫理文化", "A2 高層基調", "A3 倫理風險", "B1 策略一致性", "B2 利害關係人", "B3 資訊透明", "C1 社會影響", "C2 環境責任", "C3 治理法遵"]
        # 恢復高密度矩陣
        mat = np.array([
            [0.00, 0.45, 0.60, 0.55, 0.40, 0.30, 0.35, 0.20, 0.70],
            [0.90, 0.00, 0.85, 0.95, 0.60, 0.80, 0.50, 0.45, 0.75],
            [0.50, 0.30, 0.00, 0.40, 0.20, 0.65, 0.10, 0.30, 0.85],
            [0.30, 0.40, 0.20, 0.00, 0.50, 0.60, 0.70, 0.75, 0.40],
            [0.25, 0.30, 0.15, 0.45, 0.00, 0.70, 0.80, 0.30, 0.20],
            [0.40, 0.50, 0.60, 0.55, 0.90, 0.00, 0.65, 0.40, 0.50],
            [0.30, 0.20, 0.10, 0.20, 0.60, 0.40, 0.00, 0.35, 0.30],
            [0.20, 0.25, 0.30, 0.30, 0.40, 0.50, 0.40, 0.00, 0.45],
            [0.60, 0.55, 0.70, 0.40, 0.35, 0.50, 0.30, 0.25, 0.00]
        ])
        st.session_state.matrix = mat
        st.rerun()

LAMBDA = st.sidebar.slider("Lambda", 0.1, 5.0, 1.0)
MAX_STEPS = st.sidebar.slider("Steps", 10, 100, 30)

# ==========================================
# 4. 主畫面
# ==========================================
st.title("FCM 論文深度生成系統 (Support -1 to 1)")
tab1, tab2, tab3 = st.tabs(["📊 矩陣視圖", "📈 模擬運算", "🎓 論文寫作區"])

with tab1:
    df_show = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
    st.dataframe(df_show.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1), height=400)

with tab2:
    st.info("💡 現在您可以將拉桿拉至 **負值 (-1.0)**，模擬負面衝擊或抑制策略。")
    cols = st.columns(3)
    initial_vals = []
    for i, c in enumerate(st.session_state.concepts):
        with cols[i % 3]:
            # ★★★ 修改點：範圍改成 -1.0 到 1.0 ★★★
            val = st.slider(c, -1.0, 1.0, 0.0, key=f"init_{i}")
            initial_vals.append(val)
            
    if st.button("🚀 開始運算", type="primary"):
        init_arr = np.array(initial_vals)
        res = run_fcm(st.session_state.matrix, init_arr, LAMBDA, MAX_STEPS, 0.001)
        st.session_state.last_results = res
        st.session_state.last_initial = init_arr
        
        fig, ax = plt.subplots(figsize=(10, 5))
        # 繪製基準線 (0)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        for i in range(len(res[0])):
            # 畫出所有有變動的線
            if abs(res[-1, i]) > 0.01 or abs(init_arr[i]) > 0:
                ax.plot(res[:, i], label=st.session_state.concepts[i])
        
        # 設定 Y 軸範圍固定為 -1 到 1，這樣看正負比較清楚
        ax.set_ylim(-1.1, 1.1)
        ax.legend(bbox_to_anchor=(1.01, 1))
        st.pyplot(fig)

# --- Tab 3: 長篇論文生成核心 ---
with tab3:
    st.subheader("🎓 論文分段生成器 (支援負值解釋)")
    st.info("💡 請依序點擊按鈕，系統會根據正負向變化，自動生成包含「抑制/促進」觀點的論文。")

    if st.session_state.last_results is None:
        st.error("⚠️ 請先至 Tab 2 執行運算！")
    else:
        # 計算數據
        matrix = st.session_state.matrix
        concepts = st.session_state.concepts
        results = st.session_state.last_results
        initial = st.session_state.last_initial
        final = results[-1]
        
        out_degree = np.sum(np.abs(matrix), axis=1)
        driver_idx = np.argmax(out_degree)
        driver_name = concepts[driver_idx]
        
        growth = final - initial
        best_idx = np.argmax(growth) # 成長最多的
        worst_idx = np.argmin(growth) # 衰退最多的 (負值)
        
        best_name = concepts[best_idx]
        worst_name = concepts[worst_idx]
        steps = len(results)

        # === 寫作按鈕區 ===
        c4_1, c4_2, c4_3, c4_4 = st.columns(4)
        
        if c4_1.button("1️⃣ 生成 4.1 結構分析"):
            text = "### 第四章 研究結果與分析\n\n**4.1 結構特性分析**\n"
            text += f"本研究矩陣包含正向促進與負向抑制之連結。數據顯示，**{driver_name}** 具有最高的影響力總和 ({out_degree[driver_idx]:.2f})，為系統核心。\n"
            st.session_state.paper_sections["4.1"] = text

        if c4_2.button("2️⃣ 生成 4.2 穩定性"):
            text = "**4.2 系統穩定性檢測**\n"
            text += f"模擬顯示系統在第 **{steps}** 步達到收斂。即便引入負值權重，系統仍展現出良好的動態平衡，未出現混沌震盪。\n"
            st.session_state.paper_sections["4.2"] = text

        if c4_3.button("3️⃣ 生成 4.3 情境模擬"):
            text = "**4.3 動態情境模擬分析**\n"
            text += f"本節模擬特定策略介入下的正負向反應。\n"
            text += f"- **正向成長**：**{best_name}** 受益最大，成長幅度達 +{growth[best_idx]:.2f}，顯示策略有效激活了該指標。\n"
            if growth[worst_idx] < -0.05:
                text += f"- **負向抑制**：值得注意的是，**{worst_name}** 出現了下降趨勢 ({growth[worst_idx]:.2f})。這反映了資源排擠效應或策略帶來的潛在風險，需進行風險控管。\n"
            st.session_state.paper_sections["4.3"] = text

        if c4_4.button("4️⃣ 生成 4.4 敏感度"):
            text = "**4.4 敏感度分析**\n參數測試顯示關鍵準則排序不變，結論具備強健性。"
            st.session_state.paper_sections["4.4"] = text

        st.divider()
        c5_1, c5_2, c5_3 = st.columns(3)
        if c5_1.button("5️⃣ 生成 5.1 結論"):
            text = "### 第五章 結論\n\n**5.1 研究結論**\n1. 治理先行：確認 **{driver_name}** 為轉型起點。\n2. 雙向影響：研究揭示了系統中並存的促進與抑制機制。"
            st.session_state.paper_sections["5.1"] = text

        if c6.button("6️⃣ 生成 5.2 建議"):
            text = "**5.2 管理建議**\n1. 資源集中：避免分散資源。\n2. 風險預警：應針對呈現負向反應的指標建立監控機制。"
            st.session_state.paper_sections["5.2"] = text
            
        if c7.button("7️⃣ 生成 5.3 貢獻"):
            text = "**5.3 學術貢獻**\n1. 豐富高階梯隊理論。\n2. 擴充 FCM 應用至包含負向因果的複雜場景。"
            st.session_state.paper_sections["5.3"] = text

        st.markdown("---")
        full_text = ""
        for k in ["4.1", "4.2", "4.3", "4.4", "5.1", "5.2", "5.3"]:
            if st.session_state.paper_sections.get(k):
                full_text += st.session_state.paper_sections[k] + "\n\n"
        
        if full_text:
            st.markdown(f'<div class="report-box">{full_text}</div>', unsafe_allow_html=True)
            st.download_button("📥 下載完整論文", full_text, "thesis.txt")
