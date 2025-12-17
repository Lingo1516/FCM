import streamlit as st
import pandas as pd
import string
import re
from collections import Counter
from io import BytesIO

# --- 頁面設定 ---
st.set_page_config(page_title="智慧準則萃取器", layout="wide", page_icon="🧠")

st.title("🧠 智慧準則萃取與矩陣生成")
st.markdown("### 自動從您的資料中「提煉」出關鍵準則，並過濾掉無意義的雜字。")

# --- 1. 輸入區 ---
st.info("👇 請貼上您的文獻/筆記資料 (每一篇請記得 **按 Enter 換行**)")
raw_text = st.text_area("資料輸入區", height=250, placeholder="第一篇文獻內容...\n第二篇文獻內容...\n(不需要年份，只要換行就好)")

# --- 2. 核心：寬鬆切割邏輯 (解決找不到年份的問題) ---
def loose_parse(text):
    lines = text.strip().split('\n')
    literature_list = []
    
    for line in lines:
        line = line.strip()
        if len(line) < 4: continue # 過濾太短的
        
        # 自動抓取前15字當作代號
        name_guess = line[:15] + "..." if len(line) > 15 else line
        
        literature_list.append({
            "author": name_guess, 
            "abstract": line
        })
    return literature_list

# --- 3. 核心：改良版關鍵字算法 (解決「期下午」這種怪字) ---
def smart_keyword_extraction(text_list, top_n=15):
    # 接合所有文字
    full_text = " ".join([item['abstract'] for item in text_list])
    
    # 只留中文
    chinese_only = re.sub(r'[^\u4e00-\u9fa5]', '', full_text)
    
    words = []
    
    # --- 🔥 強力垃圾詞黑名單 (解決你的截圖問題) ---
    stop_chars = set(['的', '了', '和', '是', '就', '都', '而', '及', '與', '著', '或', '一個', '沒有', '我們', '你們', '他們', '對於', '關於', '但是', '因為', '所以', '如果', '雖然', '以及', '透過', '進行', '使用', '分析', '研究', '探討', '提出', '結果', '顯示', '發現', '本文', '摘要', '文獻', '資料', '數據', '報告'])
    
    # 針對時間日期的過濾 (解決 "日期", "期下", "下午" 等問題)
    time_words = set(['日期', '時間', '上午', '下午', '晚上', '今天', '明天', '後天', '昨天', '星期', '禮拜', '月份', '年份', '年度', '期間', '開始', '結束', '現在', '目前', '未來', '過去', '期下', '期中', '期上'])

    # 簡單切詞 (N-gram)
    for i in range(len(chinese_only)):
        # 抓 2 字詞
        if i + 2 <= len(chinese_only):
            w = chinese_only[i:i+2]
            if w not in stop_chars and w not in time_words: 
                words.append(w)
        
        # 抓 3 字詞 (準則通常是3-4字，給它多一點機會)
        if i + 3 <= len(chinese_only):
            w = chinese_only[i:i+3]
            if w not in stop_chars and w not in time_words: 
                words.append(w) # 讓它重複加入，增加權重
                words.append(w) 

        # 抓 4 字詞
        if i + 4 <= len(chinese_only):
            w = chinese_only[i:i+4]
            if w not in stop_chars and w not in time_words: 
                words.append(w)
                words.append(w) # 權重加倍

    # 統計頻率
    counter = Counter(words)
    
    # 過濾掉頻率太低(只出現一次)的雜訊
    filtered_keywords = [w for w, c in counter.most_common(50) if c > 1]
    
    # 回傳前 N 個
    return filtered_keywords[:top_n]

# --- 4. 執行與顯示 ---
if st.button("🚀 自動建構準則並分析", type="primary"):
    if not raw_text:
        st.warning("請先貼上資料！")
    else:
        # A. 切割資料
        lit_data = loose_parse(raw_text)
        
        if not lit_data:
            st.error("無法識別資料，請確認有按 Enter 換行。")
        else:
            st.success(f"✅ 成功讀取 {len(lit_data)} 筆資料，正在建構準則...")
            
            # B. 運算關鍵字
            auto_keywords = smart_keyword_extraction(lit_data)
            
            # 若運算結果很少，給予預設提示
            if not auto_keywords:
                auto_keywords = ["(資料量太少，無法統計出顯著準則，請手動輸入)"]

            # C. 讓使用者確認與修改 (這是關鍵步驟)
            st.divider()
            st.subheader("1️⃣ 系統建議的準則 (可修改)")
            
            col_sel, col_add = st.columns([2, 1])
            with col_sel:
                selected_keywords = st.multiselect(
                    "請勾選您要保留的準則：",
                    options=auto_keywords,
                    default=auto_keywords
                )
            with col_add:
                manual_add = st.text_input("手動補充準則 (用空白隔開)：", placeholder="例如：ESG 獲利能力")
            
            # 合併
            final_keywords = selected_keywords
            if manual_add:
                final_keywords = manual_add.split() + final_keywords
            
            # 去除重複
            final_keywords = list(dict.fromkeys(final_keywords))

            if final_keywords:
                # D. 生成矩陣 (跟之前一樣的邏輯)
                matrix = {}
                labels = []
                full_names = []
                
                def get_label(index):
                    if index < 26: return string.ascii_uppercase[index]
                    else: return f"{string.ascii_uppercase[index // 26 - 1]}{string.ascii_uppercase[index % 26]}"

                for i, item in enumerate(lit_data):
                    label = get_label(i)
                    labels.append(label)
                    full_names.append(item['author'])
                    
                    col_res = []
                    for kw in final_keywords:
                        if kw in item['abstract']:
                            col_res.append("○")
                        else:
                            col_res.append("")
                    matrix[label] = col_res
                
                # E. 顯示結果
                df_matrix = pd.DataFrame(matrix, index=final_keywords)
                df_legend = pd.DataFrame({"代號": labels, "對應內容": full_names})
                
                st.divider()
                st.subheader("2️⃣ 分析結果")
                
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.dataframe(df_matrix, use_container_width=True)
                with c2:
                    st.dataframe(df_legend, hide_index=True, use_container_width=True)
                
                # F. 智慧下載 (解決紅色錯誤)
                output = BytesIO()
                download_ready = False
                file_name = "matrix.csv"
                mime_type = "text/csv"
                
                try:
                    import xlsxwriter
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_matrix.to_excel(writer, sheet_name='矩陣')
                        df_legend.to_excel(writer, sheet_name='對照表')
                    file_name = "analysis.xlsx"
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    download_ready = True
                except ImportError:
                    # 如果沒裝 xlsxwriter，就下載 CSV，不報錯！
                    st.warning("⚠️ 系統偵測到未安裝 xlsxwriter，將自動改為下載 CSV 格式。")
                    output = BytesIO()
                    df_matrix.to_csv(output, encoding='utf-8-sig')
                    download_ready = True

                if download_ready:
                    st.download_button(
                        label=f"📥 下載報表 ({file_name.split('.')[-1]})", 
                        data=output.getvalue(), 
                        file_name=file_name,
                        mime=mime_type,
                        type="primary"
                    )
