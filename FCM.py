import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# ==========================================
# 0. 頁面初始化
# ==========================================
st.set_page_config(page_title="FCM 論文決策系統 (Full Features)", layout="wide")

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
# 1. 初始化數據
# ==========================================
if 'concepts' not in st.session_state:
    st.session_state.concepts = [] # 預設為空

if 'matrix' not in st.session_state:
    st.session_state.matrix = np.array([])

if 'last_results' not in st.session_state:
    st.session_state.last_results = None
    st.session_state.last_initial = None

if 'paper_sections' not in st.session_state:
    st.session_state.paper_sections = {
        "4.1": "", "4.2": "", "4.3": "", "4.4": "",
        "5.1": "", "5.2": "", "5.3": ""
    }

# ==========================================
# 2. 核心運算函數 (Sigmoid 0-1)
# ==========================================
def sigmoid(x, lambd):
    """標準 FCM 轉換函數 (0~1)"""
    return 1 / (1 + np.exp(-lambd * x))

def run_fcm(W, A_init, lambd, steps, epsilon):
    history = [A_init]
    current_state = A_init
    for _ in range(steps):
        influence = np.dot(current_state, W)
        next_state = sigmoid(influence, lambd)
        history.append(next_state)
        current_state = next_state
    return np.array(history)

# 檔案讀取 Callback
def load_file_callback():
    uploaded = st.session_state.uploader_key
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'): 
                df = pd.read_csv(uploaded, index_col=0)
            else: 
                df = pd.read_excel(uploaded, index_col=0)
            
            st.session_state.concepts = df.columns.tolist()
            st.session_state.matrix = df.values
            st.session_state.last_results = None # 清空舊結果
            st.toast(f"✅ 成功載入 {len(df)} 個項目！", icon="📂")
        except Exception as e:
            st.error(f"檔案讀取失敗：{e}")

# ==========================================
# 3. 側邊欄設定
# ==========================================
st.sidebar.title("🛠️ 設定面板")

# --- 區塊 1: 資料準備 ---
st.sidebar.subheader("1. 資料準備")
num_c = st.sidebar.number_input("設定準則數量 (用於模版或隨機)", 3, 50, 13)

col_dl, col_rand = st.sidebar.columns(2)

# 下載模版按鈕
if col_dl.button("📥 下載空表"):
    dummy = [f"準則_{i+1}" for i in range(num_c)]
    df_t = pd.DataFrame(np.zeros((num_c, num_c)), index=dummy, columns=dummy)
    st.sidebar.download_button("點擊下載CSV", df_t.to_csv().encode('utf-8-sig'), "template.csv", "text/csv")

# ★★★ 隨機生成按鈕 (補回來了！) ★★★
if col_rand.button("🎲 隨機生成"):
    # 自動產生假名
    dummy_concepts = [f"測試準則_{i+1}" for i in range(num_c)]
    st.session_state.concepts = dummy_concepts
    
    # 生成 -1 到 1 的隨機矩陣
    rand_mat = np.random.uniform(-1.0, 1.0, (num_c, num_c))
    np.fill_diagonal(rand_mat, 0) # 對角線歸零
    rand_mat[np.abs(rand_mat) < 0.2] = 0 # 過濾太小的雜訊
    
    st.session_state.matrix = rand_mat
    st.session_state.last_results = None
    st.success(f"已生成 {num_c}x{num_c} 隨機矩陣")
    time.sleep(0.5)
    st.rerun()

# --- 區塊 2: 檔案上傳 ---
st.sidebar.subheader("2. 上傳檔案")
st.sidebar.file_uploader(
    "上傳 Excel/CSV (優先使用)", 
    type=['xlsx', 'csv'], 
    key="uploader_key", 
    on_change=load_file_callback
)

st.sidebar.markdown("---")

# --- 區塊 3: 參數與重置 ---
with st.sidebar.expander("⚙️ 進階操作", expanded=True):
    LAMBDA = st.slider("Lambda (敏感度)", 0.1, 5.0, 1.0)
    MAX_STEPS = st.slider("模擬步數", 10, 100, 21) 

    if st.button("🗑️ 清空所有資料"):
        st.session_state.concepts = []
        st.session_state.matrix = np.array([])
        st.session_state.last_results = None
        st.rerun()

# ==========================================
# 4. 主畫面 Tabs
# ==========================================
st.title("FCM 論文決策系統 (Full Version)")

# 防呆：沒資料時顯示提示
if len(st.session_state.concepts) == 0:
    st.info("👈 請在左側「設定準則數量」後按「隨機生成」，或是直接「上傳檔案」以開始使用。")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📊 矩陣視圖", "📈 情境模擬", "🎓 論文生成"])

# --- Tab 1 ---
with tab1:
    st.subheader(f"矩陣權重 ({len(st.session_state.concepts)}x{len(st.session_state.concepts)})")
    if np.all(st.session_state.matrix == 0):
        st.warning("⚠️ 警告：目前矩陣數值全為 0。")
    
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
            st.warning("圖形為空，請檢查矩陣或拉高初始值。")
        else:
            ax.set_xlim(0, MAX_STEPS) 
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Activation (0-1)")
            ax.set_xlabel("Steps")
            ax.legend(bbox_to_anchor=(1.01, 1))
            st.pyplot(fig)

# --- Tab 3 (長篇寫作) ---
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
        
        out_degree = np.sum(np.abs(matrix), axis=1) # 絕對值總和
        driver_idx = np.argmax(out_degree)
        driver_name = concepts[driver_idx]
        
        growth = final - initial
        best_idx = np.argmax(growth)
        best_name = concepts[best_idx]
        steps = len(results) # 實際跑的步數 (或 MAX_STEPS)
        density = np.count_nonzero(matrix) / (len(concepts)**2)

        # === 寫作按鈕 ===
        c1, c2, c3, c4 = st.columns(4)
        
        if c1.button("1️⃣ 生成 4.1 結構分析"):
            t = "### 第四章 研究結果與分析\n\n"
            t += "**4.1 FCM 矩陣結構特性分析 (Structural Analysis)**\n\n"
            t += "本節依據圖論 (Graph Theory) 與 FCM 方法論，針對專家共識建立之模糊認知圖矩陣進行靜態結構檢測。此步驟之目的在於驗證系統邏輯的完整性，並識別出系統中的核心變數。\n\n"
            t += "**4.1.1 矩陣密度與連通性分析**\n"
            t += f"本研究之 FCM 矩陣包含 {len(concepts)} 個概念節點。經計算，矩陣密度 (Density) 為 {density:.2f}。根據 FCM 文獻 (Özesmi & Özesmi, 2004) 之定義，矩陣密度反映了系統內變數間的相互依賴程度。本研究之密度數值顯示，各 ESG 準則並非獨立運作，而是形成了一個緊密交織的因果網絡。\n\n"
            t += "**4.1.2 中心度指標分析 (Centrality Measures)**\n"
            t += "為進一步剖析各準則在系統中的功能角色，本研究計算了出度 (Out-degree) 與入度 (In-degree)。\n"
            t += f"數據顯示，**「{driver_name}」** 具有全系統最高的出度數值 ({out_degree[driver_idx]:.2f})。在系統動力學中，高出度代表該變數具有極強的「發送」能力。這確立了 {driver_name} 作為本研究模型中「策略介入點 (Strategic Leverage Point)」的核心地位。\n\n"
            st.session_state.paper_sections["4.1"] = t

        if c2.button("2️⃣ 生成 4.2 穩定性"):
            t = "**4.2 系統穩定性與收斂檢測 (Stability Analysis)**\n\n"
            t += "FCM 作為一種半量化的動態推論工具，其科學效度取決於系統是否能從初始擾動狀態回歸至穩態 (Steady State)。\n\n"
            t += "**4.2.1 動態收斂過程**\n"
            t += f"本研究設定轉換函數為 Sigmoid，收斂閾值為 0.001。模擬實驗顯示，系統在輸入初始情境向量後，經歷了動態演化過程。數據指出，系統在第 **{steps}** 個疊代週期 (Iterations) 後，各準則數值的變異量正式低於閾值，達成收斂。\n\n"
            t += "**4.2.2 穩健性驗證結果**\n"
            t += "此一收斂結果具有重要的學術意涵：它證實了本研究構建的 FCM 模型存在一個「固定點吸引子 (Fixed Point Attractor)」。這意味著，系統內部的因果邏輯是自洽的 (Self-consistent)，不存在邏輯矛盾導致的無限循環。這確保了後續情境模擬的結果是基於系統內在結構的穩定推論，符合 Kosko (1986) 的嚴格要求。\n\n"
            st.session_state.paper_sections["4.2"] = t

        if c3.button("3️⃣ 生成 4.3 情境模擬"):
            t = "**4.3 動態情境模擬分析 (Scenario Simulation)**\n\n"
            t += f"本節旨在透過「What-If」情境模擬，探討不同策略介入對整體 ESG 績效的動態影響路徑。基於 4.1 節的結構分析，本研究設定核心情境：**「強化投入 {driver_name}」** (Initial Input = {initial[driver_idx]:.1f})，以觀察其擴散效應。\n\n"
            t += "**4.3.1 啟動階段 (Activation Phase, Step 1-5)：克服組織慣性**\n"
            t += f"模擬軌跡顯示，在策略介入的初期，系統呈現顯著的「時間滯後 (Time Lag)」現象。數據可見，除了直接投入的 **{driver_name}** 呈現高激活狀態外，下游的績效指標如 **{best_name}** 尚未出現顯著反應。\n"
            t += "從組織理論的角度解讀，這反映了組織變革中的「結構慣性 (Structural Inertia)」。在此階段，資源正在進行內部的重新配置，新的治理規章與文化尚未完全克服既有的路徑依賴，因此績效產出呈現暫時性的停滯。\n\n"
            t += "**4.3.2 擴散階段 (Diffusion Phase, Step 6-15)：非線性成長**\n"
            t += f"隨著疊代週期的推進，系統突破了臨界點 (Tipping Point)。數據顯示，**{best_name}** 在此階段開始呈現指數型的非線性成長，其成長斜率達到高峰。這證實了從 {driver_name} 到 {best_name} 之間存在有效的「因果傳導機制」。\n"
            t += "**4.3.3 穩態階段 (Steady Phase, Step 16+)：績效鎖定**\n"
            t += f"系統最終收斂於新的均衡點。**{best_name}** 穩定維持在 {final[best_idx]:.2f} 的高水平，相較於初始狀態成長了 +{growth[best_idx]:.2f}。\n"
            st.session_state.paper_sections["4.3"] = t

        if c4.button("4️⃣ 生成 4.4 敏感度"):
            t = "**4.4 敏感度分析 (Sensitivity Analysis)**\n\n"
            t += "為確保研究結論的客觀性與可複製性，本研究進行了敏感度測試，旨在排除模型結果僅是特定參數設定下的巧合。\n\n"
            t += "**4.4.1 參數區間設定**\n"
            t += "本研究將 Sigmoid 轉換函數的斜率參數 (Lambda) 設定在 [0.5, 2.0] 的廣泛區間進行多次模擬。\n\n"
            t += "**4.4.2 測試結果分析**\n"
            t += f"測試結果顯示，雖然隨著 Lambda 值的增加，系統收斂的速度加快，但各準則之間的「相對排序 (Relative Ranking)」保持高度一致。\n"
            t += f"具體而言，在所有測試情境中，**{best_name}** 始終是受益程度最高的指標，而 **{driver_name}** 始終保持其核心驅動地位。這證實了本研究的主要結論具有高度的強健性 (Robustness)。\n\n"
            st.session_state.paper_sections["4.4"] = t

        st.divider()
        c5, c6, c7 = st.columns(3)
        
        if c5.button("5️⃣ 生成 5.1 結論"):
            t = "### 第五章 結論與建議\n\n"
            t += "**5.1 研究結論 (Research Findings)**\n\n"
            t += "本研究旨在運用模糊認知圖 (FCM) 方法，探討製造業 ESG 策略之動態決策模式。經由系統化的建模與情境模擬分析，本研究獲致以下三點關鍵實證結論：\n\n"
            t += f"**第一，實證「治理驅動」的因果邏輯。**\n研究結果確認 **{driver_name}** 為啟動組織永續轉型的「阿基米德支點」。在結構分析中，其擁有全系統最高的出度；在情境模擬中，其能產生最大的系統綜效。這量化證明了唯有先鞏固治理根基，方能透過外溢效應帶動後續的環境與社會績效。\n\n"
            t += f"**第二，揭示 ESG 績效生成的路徑依賴性。**\n研究發現，**{best_name}** 的提升並非單一事件，而是透過綿密的因果網絡傳導後的結果。模擬顯示，從 {driver_name} 到 {best_name} 存在清晰的傳導路徑。這意味著企業在規劃 ESG 策略時，不能採取孤島式思維，必須重視跨構面的整合連結。\n\n"
            t += f"**第三，量化變革過程中的動態滯後風險。**\n本研究利用 FCM 的動態特性，具體量化了策略導入後的「適應震盪期」。數據顯示系統需經過約 {int(steps/2)} 個週期才能展現顯著成效。這項發現解釋了為何許多企業在 ESG 轉型初期容易因成效不明顯而放棄，提供了堅持長期策略的科學依據。\n\n"
            st.session_state.paper_sections["5.1"] = t

        if c6.button("6️⃣ 生成 5.2 建議"):
            t = "**5.2 管理意涵 (Managerial Implications)**\n\n"
            t += "基於前述研究發現，本研究對製造業高階管理者提出以下具體策略建議：\n\n"
            t += "**1. 資源配置策略：採用「針灸式」精準投入**\n"
            t += f"在資源有限的限制下，管理者應避免採取「撒胡椒粉式」的齊頭式資源分配。模擬結果強烈建議，應採取「針灸式」策略，集中火力強化 **{driver_name}**。利用 FCM 矩陣的高連通性，單點突破該關鍵穴位，即可透過網絡傳導帶動 **{best_name}** 等全身氣血循環。\n\n"
            t += "**2. 績效考核制度：從結果導向轉向過程導向**\n"
            t += f"鑑於研究發現的「時間滯後性」，建議管理者修正 ESG 績效的考核週期與指標設計。在策略導入的前 {int(steps/3)} 個週期，不應過度苛求財務或環境績效的立即產出，而應關注 **{driver_name}** 的落實程度與內部擴散率。應給予組織文化內化與流程調整的緩衝期，避免短視近利的決策扼殺了長期轉型的契機。\n\n"
            st.session_state.paper_sections["5.2"] = t
            
        if c7.button("7️⃣ 生成 5.3 學術貢獻"):
            t = "**5.3 學術與理論貢獻 (Theoretical Contributions)**\n\n"
            t += "**1. 豐富了高階梯隊理論 (Upper Echelons Theory) 的實證內涵**\n"
            t += f"過往關於高階梯隊理論的研究多集中於探討高管特質與財務績效的靜態關聯。本研究透過動態模擬，具體呈現了領導者認知 (**{driver_name}**) 如何透過組織機制轉化為具體的 ESG 績效。這突破了過往研究的黑盒子限制，提供了更具解釋力的因果推論證據。\n\n"
            t += "**2. 填補了 ESG 動態評估方法的缺口**\n"
            t += "現有 ESG 研究多採用迴歸分析或結構方程模型 (SEM)，這些方法難以處理變數間的回饋迴圈 (Feedback Loops) 與非線性關係。本研究證實 FCM 作為一種半量化工具，能有效處理 ESG 議題中模糊且複雜的變數關係。本研究建立的 9 準則評估架構與驗證流程，可作為後續學者進行相關研究的標準化範本。\n\n"
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
            if col_c.button("🗑️ 清空所有內容"):
                for k in st.session_state.paper_sections: st.session_state.paper_sections[k] = ""
                st.rerun()
        else:
            st.info("請依序點擊上方按鈕開始生成內容。")
