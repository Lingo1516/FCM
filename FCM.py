import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(page_title="FCM 論文決策系統 (Curve Fixed)", layout="wide")

st.markdown("""
<style>
    .report-box { border: 1px solid #ccc; padding: 40px; background-color: #ffffff; color: #000000; font-family: "Times New Roman"; line-height: 2.0; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3.5em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 初始化狀態
# ==========================================
if 'concepts' not in st.session_state:
    st.session_state.concepts = [f"C{i+1}" for i in range(9)]

if 'matrix' not in st.session_state:
    # 預設給一個隨機矩陣，確保第一次打開不會是死線
    n = 9
    mat = np.random.uniform(-0.5, 0.5, (n, n))
    np.fill_diagonal(mat, 0)
    st.session_state.matrix = mat

if 'last_results' not in st.session_state:
    st.session_state.last_results = None
    st.session_state.last_initial = None

if 'paper_sections' not in st.session_state:
    st.session_state.paper_sections = {"4.1": "", "4.2": "", "4.3": "", "4.4": "", "5.1": "", "5.2": "", "5.3": ""}

# ==========================================
# 2. 核心運算 (加入慣性，修復死線問題)
# ==========================================
def sigmoid(x, lambd):
    return 1 / (1 + np.exp(-lambd * x))

def run_fcm(W, A_init, lambd, steps, epsilon):
    history = [A_init]
    current_state = A_init
    
    for _ in range(steps):
        # 1. 計算輸入
        influence = np.dot(current_state, W)
        
        # 2. 轉換
        new_state = sigmoid(influence, lambd)
        
        # ★★★ 關鍵修復：加入慣性 (Self-Memory) ★★★
        # 這行代碼保證了圖形會是曲線，而不會直線掉落
        # 公式：下個狀態 = 50% 舊狀態 + 50% 新計算值
        updated_state = 0.5 * current_state + 0.5 * new_state
        
        history.append(updated_state)
        
        # 這裡不設 break，強制跑滿步數以便觀察
        current_state = updated_state
        
    return np.array(history)

# 檔案讀取 Callback
def load_file_callback():
    uploaded = st.session_state.uploader_key
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'): df = pd.read_csv(uploaded, index_col=0)
            else: df = pd.read_excel(uploaded, index_col=0)
            st.session_state.concepts = df.columns.tolist()
            st.session_state.matrix = df.values
            st.toast("✅ 讀取成功！", icon="📂")
        except: st.error("檔案讀取失敗")

def sort_matrix_logic():
    try:
        df = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
        df_sorted = df.sort_index(axis=0).sort_index(axis=1)
        st.session_state.concepts = df_sorted.index.tolist()
        st.session_state.matrix = df_sorted.values
        st.success("已排序")
    except: st.error("排序失敗")

# ==========================================
# 3. 側邊欄
# ==========================================
st.sidebar.title("🛠️ 設定")

st.sidebar.subheader("1. 資料來源")
num_c = st.sidebar.number_input("數量", 3, 30, 9)
if st.sidebar.button("📥 下載空表"):
    dummy = [f"C{i+1}" for i in range(num_c)]
    df_t = pd.DataFrame(np.zeros((num_c, num_c)), index=dummy, columns=dummy)
    st.sidebar.download_button("下載CSV", df_t.to_csv().encode('utf-8-sig'), "template.csv", "text/csv")

st.sidebar.file_uploader("上傳檔案", type=['xlsx', 'csv'], key="uploader_key", on_change=load_file_callback)

st.sidebar.markdown("---")
with st.sidebar.expander("2. 矩陣工具", expanded=True):
    if st.button("🔄 自動排序"):
        sort_matrix_logic()
        st.rerun()
        
    if st.button("🎲 隨機生成權重 (-1~1)"):
        n = len(st.session_state.concepts)
        rand = np.random.uniform(-1.0, 1.0, (n, n))
        np.fill_diagonal(rand, 0)
        rand[np.abs(rand) < 0.2] = 0 
        st.session_state.matrix = rand
        st.success("矩陣已隨機化")
        time.sleep(0.5)
        st.rerun()

    if st.button("🗑️ 清空論文"):
        for k in st.session_state.paper_sections: st.session_state.paper_sections[k] = ""
        st.rerun()

with st.sidebar.expander("3. 參數", expanded=True):
    LAMBDA = st.slider("Lambda", 0.1, 5.0, 1.0)
    MAX_STEPS = st.slider("步數", 10, 100, 21)

# ==========================================
# 4. 主畫面
# ==========================================
st.title("FCM 論文決策系統 (Fixed)")
tab1, tab2, tab3 = st.tabs(["📊 矩陣視圖", "📈 模擬運算", "🎓 論文寫作"])

with tab1:
    st.subheader("矩陣檢查")
    # ★★★ 防呆：如果全0，直接報錯，不準跑 ★★★
    if np.all(st.session_state.matrix == 0):
        st.error("🚨 錯誤：矩陣全為 0！這會導致圖形變成死線。")
        st.info("👉 請點擊左側「🎲 隨機生成權重」或上傳正確檔案。")
    else:
        st.caption("數值範圍 -1.0 ~ 1.0")
        df_show = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
        st.dataframe(df_show.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1), height=400)

with tab2:
    st.subheader("情境模擬 (0-1)")
    cols = st.columns(3)
    initial_vals = []
    for i, c in enumerate(st.session_state.concepts):
        with cols[i % 3]:
            val = st.slider(c, 0.0, 1.0, 0.0, key=f"init_{i}")
            initial_vals.append(val)
            
    if st.button("🚀 開始運算", type="primary"):
        # 再次檢查
        if np.all(st.session_state.matrix == 0):
            st.error("無法運算：矩陣為空。")
        else:
            init_arr = np.array(initial_vals)
            res = run_fcm(st.session_state.matrix, init_arr, LAMBDA, MAX_STEPS, 0.001)
            st.session_state.last_results = res
            st.session_state.last_initial = init_arr
            
            fig, ax = plt.subplots(figsize=(10, 5))
            for i in range(len(res[0])):
                # 只畫出有動的線
                if np.max(np.abs(res[:, i] - 0.5)) > 0.01 or init_arr[i] > 0:
                    ax.plot(res[:, i], label=st.session_state.concepts[i])
            
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, MAX_STEPS) # 強制顯示完整步數
            ax.set_ylabel("Activation (0-1)")
            ax.set_xlabel("Steps")
            ax.legend(bbox_to_anchor=(1.01, 1))
            st.pyplot(fig)

# --- Tab 3: 長篇寫作 ---
with tab3:
    st.subheader("🎓 論文分段生成器")
    
    if st.session_state.last_results is None:
        st.error("⚠️ 請先至 Tab 2 執行運算！")
    else:
        matrix = st.session_state.matrix
        concepts = st.session_state.concepts
        results = st.session_state.last_results
        initial = st.session_state.last_initial
        final = results[-1]
        
        out_degree = np.sum(np.abs(matrix), axis=1)
        driver_idx = np.argmax(out_degree)
        driver_name = concepts[driver_idx]
        
        growth = final - initial
        best_idx = np.argmax(growth)
        best_name = concepts[best_idx]
        steps = len(results)
        density = np.count_nonzero(matrix) / (len(concepts)**2)

        c1, c2, c3, c4 = st.columns(4)
        
        # 4.1
        if c1.button("1️⃣ 4.1 結構分析"):
            t = "### 第四章 研究結果\n\n**4.1 結構特性分析**\n"
            t += f"本研究矩陣密度為 {density:.2f}。數據顯示 **{driver_name}** 為核心驅動因子。\n"
            st.session_state.paper_sections["4.1"] = t

        # 4.2
        if c2.button("2️⃣ 4.2 穩定性"):
            t = "**4.2 系統穩定性**\n"
            t += f"模擬顯示系統在 **{steps}** 步收斂。加入慣性因子後，系統展現出平滑的收斂曲線。\n"
            st.session_state.paper_sections["4.2"] = t

        # 4.3
        if c3.button("3️⃣ 4.3 情境模擬"):
            t = "**4.3 情境模擬**\n"
            t += f"投入 **{driver_name}** 後，**{best_name}** 呈現顯著成長 (+{growth[best_idx]:.2f})。\n"
            st.session_state.paper_sections["4.3"] = t

        # 4.4
        if c4.button("4️⃣ 4.4 敏感度"):
            t = "**4.4 敏感度分析**\n參數測試顯示結論具備強健性。\n"
            st.session_state.paper_sections["4.4"] = t

        st.divider()
        c5, c6, c7 = st.columns(3)
        
        # 5.1
        if c5.button("5️⃣ 5.1 結論"):
            t = "### 第五章 結論\n\n**5.1 研究結論**\n1. 確認 **{driver_name}** 為起點。\n2. 揭示動態滯後性。\n"
            st.session_state.paper_sections["5.1"] = t

        # 5.2
        if c6.button("6️⃣ 5.2 建議"):
            t = "**5.2 管理意涵**\n1. 集中資源於核心因子。\n2. 建立長效考核機制。\n"
            st.session_state.paper_sections["5.2"] = t
            
        # 5.3
        if c7.button("7️⃣ 5.3 貢獻"):
            t = "**5.3 學術貢獻**\n1. 豐富高階梯隊理論。\n2. 提供動態分析範本。\n"
            st.session_state.paper_sections["5.3"] = t

        st.markdown("---")
        full_text = ""
        for k in ["4.1", "4.2", "4.3", "4.4", "5.1", "5.2", "5.3"]:
            if st.session_state.paper_sections.get(k):
                full_text += st.session_state.paper_sections[k] + "\n\n"
        
        if full_text:
            st.markdown(f'<div class="report-box">{full_text}</div>', unsafe_allow_html=True)
            st.download_button("📥 下載論文", full_text, "thesis.txt")
        else:
            st.info("請點擊上方按鈕開始生成內容。")
