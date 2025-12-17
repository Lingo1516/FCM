import streamlit as st
import pandas as pd
import re
import string
from collections import Counter
from io import BytesIO

# --- 頁面設定 ---
st.set_page_config(page_title="免安裝全自動分析器", layout="wide", page_icon="⚡")

st.title("⚡ 免安裝・全自動文獻分析器")
st.markdown("### 不需安裝 jieba，直接使用原生 Python 運算")

# --- 輸入區 ---
raw_text = st.text_area("👉 請貼上亂亂的文獻資料：", height=300, placeholder="直接把摘要全部貼進來...")

# --- 核心邏輯：手刻一個簡單的斷詞器 ---
def simple_keyword_extraction(text, top_n=20):
    # 1. 只保留中文字 (過濾掉標點符號和英文)
    # 這是為了讓統計更準確
    chinese_only = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
    
    # 2. 建立 n-gram (雙字詞與三字詞)
    # 因為我們沒有 jieba，所以我們假設「兩個字」或「三個字」連在一起出現最多次的，就是關鍵字
    words = []
    content = chinese_only.split()
    
    # 定義一些廢話 (Stopwords)，不要讓它們變成關鍵字
    stopwords = set(['研究', '本研究', '分析', '探討', '結果', '顯示', '發現', '提出', '認為', '使用', '進行', '影響', '我們', '這些', '不同', '以及', '透過', '對於'])

    for part in content:
        if len(part) < 2: continue
        
        # 抓取 2 個字的詞 (Bigrams)
        for i in range(len(part) - 1):
            w = part[i:i+2]
            if w not in stopwords: words.append(w)
            
        # 抓取 3 個字的詞 (Trigrams) - 權重高一點
        for i in range(len(part) - 2):
            w = part[i:i+3]
            if w not in stopwords: words.append(w)
            
    # 3. 統計出現頻率最高的詞
    counter = Counter(words)
    most_common = [w for w, c in counter.most_common(top_n)]
    
    return most_common

# --- 解析文獻結構 ---
def parse_literature(text):
    # 抓年份 (20xx)
    pattern = r'([^\n\r。]+?[\(\[\{](?:19|20)\d{2}[\)\]\}])'
    segments = re.split(pattern, text)
    
    literature_list = []
    current_author = None
    all_content_for_analysis = ""
    
    for segment in segments:
        segment = segment.strip()
        if not segment: continue
        
        if re.search(r'[\(\[\{](?:19|20)\d{2}[\)\]\}]', segment):
            current_author = segment[-50:]
        else:
            if current_author:
                literature_list.append({"author": current_author, "abstract": segment})
                all_content_for_analysis += segment + "\n"
                
    return literature_list, all_content_for_analysis

# --- 執行按鈕 ---
if st.button("🚀 自動分析 (免安裝版)", type="primary"):
    if not raw_text:
        st.warning("請先貼上資料！")
    else:
        # 1. 解析結構
        lit_data, full_text = parse_literature(raw_text)
        
        if not lit_data:
            st.error("找不到年份特徵 (例如: 2023)，無法切分文獻。")
        else:
            # 2. 執行免安裝的關鍵字抓取
            auto_keywords = simple_keyword_extraction(full_text)
            
            st.success(f"✅ 分析完成！系統用統計法抓到了 {len(auto_keywords)} 個高頻詞：")
            
            # 3. 讓使用者篩選
            final_keywords = st.multiselect(
                "系統自動抓到的關鍵字 (可刪除不準的)",
                options=auto_keywords,
                default=auto_keywords
            )
            
            if final_keywords:
                # 4. 畫圖
                matrix = {}
                labels = []
                authors = []
                
                # 生成 A, B, C
                def get_label(index):
                    if index < 26: return string.ascii_uppercase[index]
                    else: return f"{string.ascii_uppercase[index // 26 - 1]}{string.ascii_uppercase[index % 26]}"

                for i, item in enumerate(lit_data):
                    label = get_label(i)
                    labels.append(label)
                    authors.append(item['author'])
                    
                    col_res = []
                    for kw in final_keywords:
                        if kw in item['abstract']:
                            col_res.append("●")
                        else:
                            col_res.append("")
                    matrix[label] = col_res
                
                # 顯示表格
                df_matrix = pd.DataFrame(matrix, index=final_keywords)
                df_legend = pd.DataFrame({"代號": labels, "作者": authors})
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(df_matrix, use_container_width=True)
                with col2:
                    st.dataframe(df_legend, hide_index=True, use_container_width=True)
                
                # 下載
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_matrix.to_excel(writer, sheet_name='矩陣')
                    df_legend.to_excel(writer, sheet_name='作者')
                st.download_button("📥 下載 Excel", output.getvalue(), "no_install_analysis.xlsx")
