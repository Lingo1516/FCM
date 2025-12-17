import streamlit as st
import pandas as pd
import string
from collections import Counter
from io import BytesIO
import re

# --- 頁面設定 ---
st.set_page_config(page_title="萬能文獻矩陣生成器", layout="wide", page_icon="🛡️")

st.title("🛡️ 萬能文獻矩陣生成器 (容錯版)")
st.markdown("""
### 解決無法辨識的問題：
* **不再強制要求年份**：只要你的資料有 **「換行」**，程式就會自動切分。
* **自動抓取關鍵字**：運用統計原理，自動找出出現最多次的詞。
""")

# --- 1. 輸入區 ---
st.info("👇 請將文獻資料貼在下方 (請確保每一篇文獻都在 **新的一行**)")
raw_text = st.text_area("文獻資料輸入區：", height=250, placeholder="例如：\n第一篇文獻的摘要內容...\n第二篇關於績效管理的內容...\n第三篇討論員工滿意度的...")

# --- 2. 核心邏輯：最簡單暴力的切割法 ---
def loose_parse(text):
    # 直接用「換行符號」來切割，不管內容是什麼
    lines = text.strip().split('\n')
    
    literature_list = []
    
    for line in lines:
        line = line.strip()
        if len(line) < 5: continue # 過濾掉太短的廢話或空行
        
        # 嘗試自動抓一個「標題」或「作者」給它
        # 邏輯：取這行文字的前 15 個字當作代號名稱
        author_guess = line[:15] + "..." if len(line) > 15 else line
        
        literature_list.append({
            "author": author_guess, # 這是給對照表用的名稱
            "abstract": line        # 這是要拿來分析的內容
        })
        
    return literature_list

# --- 3. 免安裝關鍵字統計 (N-gram) ---
def simple_keyword_extraction(text_list, top_n=20):
    # 把所有文獻串在一起分析
    full_text = " ".join([item['abstract'] for item in text_list])
    
    # 只保留中文 (讓統計更準)
    chinese_only = re.sub(r'[^\u4e00-\u9fa5]', '', full_text)
    
    words = []
    # 排除常見無意義詞彙
    stopwords = set(['研究', '探討', '分析', '結果', '顯示', '發現', '提出', '認為', '使用', '進行', '影響', '我們', '這些', '不同', '以及', '透過', '對於', '文獻', '本文'])

    # 抓取 2~4 個字的詞
    for i in range(len(chinese_only)):
        # 2字詞
        if i + 2 <= len(chinese_only):
            w = chinese_only[i:i+2]
            if w not in stopwords: words.append(w)
        # 3字詞
        if i + 3 <= len(chinese_only):
            w = chinese_only[i:i+3]
            if w not in stopwords: words.append(w)
        # 4字詞
        if i + 4 <= len(chinese_only):
            w = chinese_only[i:i+4]
            if w not in stopwords: words.append(w)
            
    # 統計頻率
    counter = Counter(words)
    # 取出前 N 個高頻詞
    most_common = [w for w, c in counter.most_common(top_n)]
    return most_common

# --- 4. 執行按鈕 ---
col_action, col_manual = st.columns([1, 2])

with col_action:
    run_btn = st.button("🚀 強制分析", type="primary", help="不管格式對不對，按下去就對了")

# 這裡預留一個 Session State 存關鍵字，以免重整後不見
if 'auto_keywords' not in st.session_state:
    st.session_state.auto_keywords = []

if run_btn and raw_text:
    # 1. 切割文獻
    lit_data = loose_parse(raw_text)
    
    if not lit_data:
        st.error("❌ 無法切割資料。請確認你有貼上文字，而且有按 Enter 換行。")
    else:
        st.success(f"✅ 成功切分出 {len(lit_data)} 筆資料！")
        
        # 2. 抓關鍵字
        st.session_state.auto_keywords = simple_keyword_extraction(lit_data)

# --- 5. 結果顯示與篩選 ---
if st.session_state.auto_keywords:
    
    st.divider()
    st.subheader("1️⃣ 確認分析準則 (關鍵字)")
    
    # 讓使用者可以自己加關鍵字！這很重要，因為自動抓的不一定準
    user_added = st.text_input("想要手動增加關鍵字嗎？(用空白隔開)", placeholder="例如：ESG 數位轉型")
    
    # 合併自動抓的 + 手動加的
    all_options = st.session_state.auto_keywords
    if user_added:
        extras = user_added.split()
        all_options = extras + all_options
    
    final_keywords = st.multiselect(
        "請勾選您要顯示在表格左側的準則：",
        options=all_options,
        default=all_options[:10] # 預設只選前10個避免表格太大
    )
    
    if final_keywords:
        # 3. 重新切割一次以確保資料最新 (或直接用上面的 lit_data 若想優化效能)
        lit_data = loose_parse(raw_text)
        
        # 4. 製作矩陣
        matrix = {}
        labels = []
        full_names = []
        
        # 生成代號 A, B, C...
        def get_label(index):
            if index < 26: return string.ascii_uppercase[index]
            else: return f"{string.ascii_uppercase[index // 26 - 1]}{string.ascii_uppercase[index % 26]}"

        for i, item in enumerate(lit_data):
            label = get_label(i)
            labels.append(label)
            full_names.append(item['author']) # 對照表用的完整名稱
            
            # 比對
            col_res = []
            for kw in final_keywords:
                if kw in item['abstract']:
                    col_res.append("○") # 符合你的圖片格式
                else:
                    col_res.append("")
            matrix[label] = col_res
            
        # 轉 DataFrame
        df_matrix = pd.DataFrame(matrix, index=final_keywords)
        df_matrix.index.name = "構面\\準則"
        
        df_legend = pd.DataFrame({
            "文獻標籤": labels,
            "對應內容 (前15字)": full_names
        })
        
        # --- 顯示最終結果 (模仿你的截圖) ---
        st.divider()
        st.subheader("2️⃣ 分析結果矩陣")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("##### 重新整理後的表格格式：")
            st.dataframe(df_matrix, use_container_width=True)
            
        with col2:
            st.markdown("##### 最底部的作者對應表：")
            st.dataframe(df_legend, hide_index=True, use_container_width=True)
            
        # --- 下載 ---
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_matrix.to_excel(writer, sheet_name='矩陣分析')
            df_legend.to_excel(writer, sheet_name='對照表')
            
        st.download_button(
            "📥 下載 Excel 檔案", 
            data=output.getvalue(), 
            file_name="analysis_result.xlsx", 
            type="primary"
        )
