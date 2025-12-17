import streamlit as st
import pandas as pd
import requests
import string
import re
from collections import Counter
from io import BytesIO

# --- 嘗試匯入套件 ---
try:
    import xlsxwriter
    import jieba
    import jieba.analyse
except ImportError:
    pass

# --- 1. 設定 API Key ---
USER_API_KEY = "AIzaSyBlj24gBVr3RJhkukS9p6yo5s2-WVBH2H0" 

# --- 2. 頁面設定 ---
st.set_page_config(page_title="AI 雙引擎文獻分析", layout="wide", page_icon="🛡️")
st.title("🛡️ AI 雙引擎文獻分析器 (永不當機版)")
st.markdown("### 邏輯：優先使用 Google AI，若額度額滿(429)則自動切換至本機演算法。")

# --- 3. 核心：Google AI 分析 ---
def analyze_with_google(text, key):
    # 使用 gemini-1.5-flash (免費額度較高)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
    headers = {'Content-Type': 'application/json'}
    prompt = f"任務：歸納 10 個學術研究構面關鍵字。規則：只列出名詞，用頓號隔開。排除無關詞彙(如日期、下午)。內容：{text[:5000]}"
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return "SUCCESS", response.json()['candidates'][0]['content']['parts'][0]['text']
        elif response.status_code == 429:
            return "QUOTA_ERROR", "額度額滿"
        else:
            return "OTHER_ERROR", response.text
    except Exception as e:
        return "NET_ERROR", str(e)

# --- 4. 核心：本機備用演算法 (Jieba) ---
def analyze_with_local(text):
    # 這是備案，當 AI 掛掉時使用
    # 1. 嘗試用 jieba 抓關鍵字
    try:
        keywords = jieba.analyse.extract_tags(text, topK=15, allowPOS=('n', 'vn', 'v'))
        return keywords
    except:
        # 萬一連 jieba 都沒裝，用最笨的方法切
        clean_text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        words = [clean_text[i:i+2] for i in range(len(clean_text)-1)]
        return [w for w, c in Counter(words).most_common(15)]

# --- 5. 切割與執行 ---
st.info("👇 請貼上文獻資料 (每篇換行)")
raw_text = st.text_area("文獻輸入", height=200)

def parse_text(text):
    lines = text.strip().split('\n')
    return [{"title": line[:15], "content": line} for line in lines if len(line) > 5]

if st.button("🚀 開始智慧分析", type="primary"):
    if not raw_text:
        st.warning("請先貼上資料")
    else:
        status_msg = st.empty()
        status_msg.info("🤖 正在嘗試呼叫 Google AI...")
        
        # 1. 先試試看 Google
        status, result_text = analyze_with_google(raw_text, USER_API_KEY)
        
        final_keywords = []
        source_used = ""
        
        if status == "SUCCESS":
            status_msg.success("✅ Google AI 連線成功！")
            final_keywords = [k.strip() for k in result_text.replace("\n", "、").split("、") if k.strip()]
            source_used = "Google AI"
        
        else:
            # 2. 如果 Google 失敗 (429 或其他)，啟動備用方案
            error_reason = "額度已滿 (429)" if status == "QUOTA_ERROR" else "連線問題"
            status_msg.warning(f"⚠️ Google AI 暫時無法使用 ({error_reason})，已自動切換至「本機演算法」繼續分析...")
            final_keywords = analyze_with_local(raw_text)
            source_used = "本機演算法 (備用模式)"
            
        # --- 下面是製表 (不管用哪種方法，這裡都會執行) ---
        st.divider()
        st.markdown(f"**本次分析來源：{source_used}**")
        
        # 讓使用者篩選
        selected_keywords = st.multiselect("分析準則 (可刪減)", options=final_keywords, default=final_keywords)
        
        if selected_keywords:
            lit_data = parse_text(raw_text)
            matrix = {}
            labels = []
            titles = []
            
            for i, item in enumerate(lit_data):
                lbl = string.ascii_uppercase[i % 26]
                labels.append(lbl)
                titles.append(item['title'])
                col_res = []
                for kw in selected_keywords:
                    if kw in item['content']: col_res.append("○")
                    else: col_res.append("")
                matrix[lbl] = col_res
            
            df = pd.DataFrame(matrix, index=selected_keywords)
            df_legend = pd.DataFrame({"代號": labels, "文獻": titles})
            
            c1, c2 = st.columns([2, 1])
            with c1: st.dataframe(df, use_container_width=True)
            with c2: st.dataframe(df_legend, hide_index=True)
            
            # 下載
            output = BytesIO()
            try:
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='矩陣')
                    df_legend.to_excel(writer, sheet_name='對照表')
                st.download_button("📥 下載 Excel", output.getvalue(), "analysis.xlsx")
            except:
                st.download_button("📥 下載 CSV", df.to_csv().encode('utf-8-sig'), "analysis.csv")
