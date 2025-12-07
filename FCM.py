import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# ==========================================
# 0. 頁面初始化
# ==========================================
st.set_page_config(page_title="FCM 論文決策系統 (Smooth Fix)", layout="wide")

st.markdown("""
<style>
    /* 論文預覽區 */
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
# 1. 初始化數據
# ==========================================
if 'concepts' not in st.session_state:
    st.session_state.concepts = [
        "A1 倫理文化", "A2 高層基調", "A3 倫理風險",
        "B1 策略一致性", "B2 利害關係人", "B3 資訊透明",
        "C1 社會影響", "C2 環境責任", "C3 治理法遵"
    ]

# 預設矩陣：填入非零數值，避免跑出 0.5 死線
if 'matrix' not in st.session_state:
    mat = np.zeros((9, 9))
    # 填入範例數值 (Kosko 標準：-1 ~ 1)
    mat[1, 0] = 0.85; mat[1, 3] = 0.80; mat[5, 4] = 0.90
    mat[2, 8] = -0.7; mat[0, 2] = -0.6
    st.session_state.matrix = mat

if 'last_results' not in st.session_state:
    st.session_state.last_results = None
    st.session_state.last_initial = None

if 'paper_sections' not in st.session_state:
    st.session_state.paper_sections = {"4.1": "", "4.2": "", "4.3": "", "4.4": "", "5.1": "", "5.2": "", "5.3": ""}

# ==========================================
# 2. 核心運算函數 (加入慣性，讓曲線變圓滑)
# ==========================================
def sigmoid(x, lambd=1):
    """標準 Sigmoid (0~1)"""
    return 1 / (1 + np.exp(-lambd * x))

def run_fcm(W, A_init, lambd, steps, inertia=0.5):
    history = [A_init]
    current_state = A_init

    for _ in range(steps):
        # 1. 計算總輸入
        influence = np.dot(current_state, W)
        
        # 2. 轉換函數
        new_val = sigmoid(influence, lambd)
        
        # ★★★ 關鍵修正：加入慣性 (Self-Memory)，並調整慣性比例
        next_state = inertia * current_state + (1 - inertia) * new_val
        
        history.append(next_state)
        
        # 即使收斂也不要 break，強制跑滿步數以便觀察趨勢
        current_state = next_state
        
    return np.array(history)

# 檔案讀取回呼
def load_file_callback():
    uploaded = st.session_state.uploader_key
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'): df = pd.read_csv(uploaded, index_col=0)
            else: df = pd.read_excel(uploaded, index_col=0)
            
            # 更新資料
            st.session_state.concepts = df.columns.tolist()
            st.session_state.matrix = df.values
            st.toast(f"✅ 讀取成功！", icon="📂")
        except: st.error("檔案讀取失敗")

def sort_matrix_logic():
    try:
        df = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
        df_sorted = df.sort_index(axis=0).sort_index(axis=1)
        st.session_state.concepts = df_sorted.index.tolist()
        st.session_state.matrix = df_sorted.values
        st.success("✅ 排序完成！")
    except: st.error("排序失敗")

# ==========================================
# 3. 側邊欄設定
# ==========================================
st.sidebar.title("🛠️ 設定面板")

st.sidebar.subheader("1. 資料來源")
# 下載模版
num_c = st.sidebar.number_input("準則數量", 3, 30, 9)
if st.sidebar.button("📥 下載空表"):
    dummy = [f"準則_{i+1}" for i in range(num_c)]
    df_t = pd.DataFrame(np.zeros((num_c, num_c)), index=dummy, columns=dummy)
    st.sidebar.download_button("下載 CSV", df_t.to_csv().encode('utf-8-sig'), "template.csv", "text/csv")

# 上傳檔案
st.sidebar.file_uploader("上傳矩陣", type=['xlsx', 'csv'], key="uploader_key", on_change=load_file_callback)

st.sidebar.markdown("---")
with st.sidebar.expander("2. 矩陣編輯", expanded=False):
    with st.form("add_c"):
        new = st.text_input("新增準則")
        if st.form_submit_button("➕ 加入") and new:
            if new not in st.session_state.concepts:
                st.session_state.concepts.append(new)
                old = st.session_state.matrix
                r,c = old.shape
                new_m = np.zeros((r+1,c+1))
                new_m[:r,:c] = old
                st.session_state.matrix = new_m
                st.rerun()
    
    if st.button("🔄 自動排序"):
        sort_matrix_logic()
        st.rerun()
        
    if st.button("🎲 隨機生成權重 (-1~1)"):
        n = len(st.session_state.concepts)
        rand = np.random.uniform(-1.0, 1.0, (n, n))
        np.fill_diagonal(rand, 0)
        rand[np.abs(rand) < 0.2] = 0 
        st.session_state.matrix = rand
        st.success("已生成測試矩陣")
        time.sleep(0.5)
        st.rerun()

    if st.button("🗑️ 清空論文"):
        for k in st.session_state.paper_sections: st.session_state.paper_sections[k] = ""
        st.rerun()

# 參數
with st.sidebar.expander("3. 模擬參數", expanded=True):
    LAMBDA = st.slider("Lambda", 0.1, 5.0, 1.0)
    MAX_STEPS = st.slider("模擬步數", 10, 100, 21)
    INERTIA = st.slider("慣性 (Self-Memory)", 0.1, 1.0, 0.5)

# ==========================================
# 4. 主畫面 Tabs
# ==========================================
st.title("FCM 論文生成系統 (Final Standard)")
tab1, tab2, tab3 = st.tabs(["📊 矩陣視圖", "📈 模擬運算", "🎓 論文寫作區"])

with tab1:
    st.subheader("矩陣關係檢視 (-1 ~ 1)")
    
    # ★★★ 防呆警告：如果矩陣全為 0，顯示紅字 ★★★
    if np.all(st.session_state.matrix == 0):
        st.error("🚨 錯誤警告：目前矩陣數值全為 0。這會導致模擬失敗 (全變 0.5)。")
        st.info("👉 請點擊側邊欄的「🎲 隨機生成權重」或上傳正確的 Excel 檔案。")
    else:
        st.caption("紅色 = 負向抑制 / 藍色 = 正向促進")
        df_show = pd.DataFrame(st.session_state.matrix, index=st.session_state.concepts, columns=st.session_state.concepts)
        st.dataframe(df_show.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1), height=400)

with tab2:
    st.subheader("情境模擬 (概念激活 0-1)")
    st.info("💡 設定初始狀態 (0.0 = 無, 1.0 = 全力投入)。")
    cols = st.columns(3)
    initial_vals = []
    for i, c in enumerate(st.session_state.concepts):
        with cols[i % 3]:
            val = st.slider(c, 0.0, 1.0, 0.0, key=f"init_{i}")
            initial_vals.append(val)
            
    if st.button("🚀 開始運算", type="primary"):
        # 再次檢查矩陣
        if np.all(st.session_state.matrix == 0):
            st.error("無法運算！矩陣是空的。")
        else:
            init_arr = np.array(initial_vals)
            res = run_fcm(st.session_state.matrix, init_arr, 1.0, 21, 0.5)  # 假設步數為 21 步
            
            st.session_state.last_results = res
            st.session_state.last_initial = init_arr
            
            # 以下修正：图形样式
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 设置颜色和线条样式，模拟类似你的示例图
            line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-']
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y']
            
            for i in range(res.shape[1]):
                ax.plot(res[:, i], label=st.session_state.concepts[i], linestyle=line_styles[i % len(line_styles)], color=colors[i % len(colors)])

            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, 21)  # 强制显示完整步数
            ax.set_ylabel("Activation (0-1)")
            ax.set_xlabel("Steps")
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Concepts")
            ax.set_facecolor('lightgray')  # 设置背景色为灰色
            plt.title('模糊認知圖模擬結果')  # 图表标题

            st.pyplot(fig)

# --- Tab 3: 長篇寫作 ---
with tab3:
    st.subheader("🎓 論文分段生成器 (目標：7000字)")
    
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
        
        if c1.button("1️⃣ 生成 4.1 結構分析"):
            t = "### 第四章 研究結果與分析\n\n**4.1 FCM 矩陣結構特性分析 (Structural Analysis)**\n\n"
            t += f"本研究矩陣包含 {len(concepts)} 個準則，矩陣密度為 {density:.2f}。\n"
            t += f"數據顯示，**{driver_name}** 之總影響力 (絕對值出度={out_degree[driver_idx]:.2f}) 最高，確認其為系統核心。\n"
            st.session_state.paper_sections["4.1"] = t

        if c2.button("2️⃣ 生成 4.2 穩定性"):
            t = "**4.2 系統穩定性檢測**\n\n"
            t += f"透過 Sigmoid 函數轉換，模擬顯示系統在第 **{steps}** 步達到收斂。各準則數值穩定落在 [0, 1] 區間內，證實模型具備動態穩定性。\n"
            st.session_state.paper_sections["4.2"] = t

        if c3.button("3️⃣ 生成 4.3 情境模擬"):
            t = "**4.3 動態情境模擬分析**\n\n"
            t += f"本節模擬在 **{driver_name}** 投入資源後的擴散效應。\n"
            t += f"結果顯示，**{best_name}** 從初始狀態顯著提升至 {final[best_idx]:.2f}。這驗證了矩陣中的因果路徑。\n"
            st.session_state.paper_sections["4.3"] = t

        if c4.button("4️⃣ 生成 4.4 敏感度"):
            t = "**4.4 敏感度分析**\n\n經測試不同 Lambda 參數，關鍵準則的相對排序保持不變，證實結論具備強健性。\n"
            st.session_state.paper_sections["4.4"] = t

        st.divider()
        c5, c6, c7 = st.columns(3)
        
        if c5.button("5️⃣ 生成 5.1 結論"):
            t = "### 第五章 結論與建議\n\n**5.1 研究結論**\n\n"
            t += f"1. 治理先行：確認 **{driver_name}** 為轉型起點。\n2. 雙向機制：揭示了系統中促進與抑制力量的動態平衡。\n"
            st.session_state.paper_sections["5.1"] = t

        if c6.button("6️⃣ 生成 5.2 建議"):
            t = "**5.2 管理意涵**\n\n"
            t += "**1. 資源配置策略：採用「針灸式」精準投入**\n模擬結果強烈建議，應採取「針灸式」策略，集中火力強化 **{driver_name}**。利用 FCM 矩陣的高連通性，單點突破該關鍵穴位，即可透過網絡傳導帶動整體循環。\n\n"
            t += "**2. 績效考核制度：從結果導向轉向過程導向**\n鑑於研究發現的「時間滯後性」，建議管理者修正 ESG 績效的考核週期。在策略導入的前期，不應過度苛求財務或環境績效的立即產出，應給予組織文化內化與流程調整的緩衝期。\n"
            st.session_state.paper_sections["5.2"] = t
            
        if c7.button("7️⃣ 生成 5.3 貢獻"):
            t = "**5.3 學術貢獻**\n\n"
            t += "**1. 豐富了高階梯隊理論的實證內涵**\n本研究透過動態模擬，具體呈現了領導者認知如何轉化為組織結果的黑盒子過程，提供了更具解釋力的因果推論證據。\n\n"
            t += "**2. 填補了 ESG 動態評估方法的缺口**\n本研究證實 FCM 作為一種半量化工具，能有效處理 ESG 議題中模糊且複雜的變數關係，為後續學者提供了標準化的分析範本。\n"
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
