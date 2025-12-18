import streamlit as st
import requests
import pandas as pd
import string
from io import BytesIO

# --- 嘗試匯入備用套件 ---
try:
    import xlsxwriter
except ImportError:
    pass

# --- 1. 設定 API Key ---
USER_API_KEY = "AIzaSyBlj24gBVr3RJhkukS9p6yo5s2-WVBH2H0" 

st.set_page_config(page_title="API 金鑰聽診器", layout="wide", page_icon="🩺")

st.title("🩺 Google API 金鑰診斷室")
st.markdown("### 讓我們找出為什麼所有模型都顯示 404 的真正原因")

if st.button("🚀 開始診斷", type="primary"):
    st.divider()
    
    # --- 測試 1: 檢查鑰匙是否有效 (ListModels) ---
    st.subheader("第一關：檢查鑰匙權限")
    list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={USER_API_KEY}"
    
    try:
        response = requests.get(list_url)
        
        if response.status_code == 200:
            st.success("✅ 第一關通過：金鑰有效，可以連線到 Google！")
            models = response.json().get('models', [])
            
            # 篩選出能用的 Gemini 模型
            gemini_models = [m['name'] for m in models if 'generateContent' in m.get('supportedGenerationMethods', []) and 'gemini' in m['name']]
            
            if gemini_models:
                st.info(f"📋 您的金鑰目前可以看到 {len(gemini_models)} 個模型：")
                st.code(gemini_models)
                
                # --- 測試 2: 實際寫入測試 (GenerateContent) ---
                st.subheader("第二關：寫入測試 (Hello World)")
                
                # 自動挑選第一個模型來測，不手動指定，避免拼錯
                test_model = gemini_models[0] 
                st.write(f"正在嘗試使用清單中的第一個模型：`{test_model}` 進行測試...")
                
                gen_url = f"https://generativelanguage.googleapis.com/v1beta/{test_model}:generateContent?key={USER_API_KEY}"
                headers = {'Content-Type': 'application/json'}
                data = {"contents": [{"parts": [{"text": "Hi"}]}]}
                
                test_resp = requests.post(gen_url, headers=headers, json=data)
                
                if test_resp.status_code == 200:
                    st.success(f"🎉 恭喜！診斷完成，模型 `{test_model}` 運作正常！")
                    st.balloons()
                    # 只有通過測試，才把這個模型存起來給下面用
                    st.session_state.valid_model = test_model
                else:
                    st.error(f"❌ 第二關失敗！雖然看得到模型，但無法使用。")
                    st.code(f"錯誤代碼: {test_resp.status_code}\n錯誤訊息: {test_resp.text}")
                    st.warning("推測原因：您的 Google Cloud 專案可能未啟用 'Generative AI API'，或者該模型在此地區不可用。")
            else:
                st.error("❌ 找不到任何 Gemini 模型！您的金鑰權限可能被嚴重限制。")
                
        else:
            st.error("❌ 第一關就失敗了：無法獲取模型清單。")
            st.code(f"錯誤代碼: {response.status_code}\n錯誤訊息: {response.text}")
            if response.status_code == 400:
                st.warning("⚠️ 診斷：金鑰格式錯誤 (Key Invalid)。")
            elif response.status_code == 403:
                st.warning("⚠️ 診斷：金鑰權限不足 (Permission Denied)。")

    except Exception as e:
        st.error(f"連線發生意外錯誤：{e}")

# --- 只有診斷成功才會顯示分析介面 ---
if 'valid_model' in st.session_state:
    st.divider()
    st.header("📄 文獻分析工作區 (已修復)")
    st.success(f"目前使用經診斷確認可用的模型：**{st.session_state.valid_model}**")
    
    raw_text = st.text_area("請輸入資料：", height=200)
    
    if st.button("開始分析"):
        if not raw_text:
            st.warning("請輸入內容")
        else:
            # 直接使用剛剛診斷成功的那個模型網址 (最穩)
            target_model = st.session_state.valid_model
            # 注意：這裡 target_model 已經包含 'models/' 前綴，不需要再加
            if not target_model.startswith("models/"):
                 target_model = f"models/{target_model}"

            url = f"https://generativelanguage.googleapis.com/v1beta/{target_model}:generateContent?key={USER_API_KEY}"
            headers = {'Content-Type': 'application/json'}
            prompt = f"歸納10個學術構面名詞，用頓號隔開：{raw_text[:5000]}"
            data = {"contents": [{"parts": [{"text": prompt}]}]}
            
            try:
                r = requests.post(url, headers=headers, json=data)
                if r.status_code == 200:
                    res = r.json()['candidates'][0]['content']['parts'][0]['text']
                    keywords = [k.strip() for k in res.replace("\n", "、").split("、") if k.strip()]
                    st.multiselect("分析結果", options=keywords, default=keywords)
                else:
                    st.error(f"分析失敗: {r.text}")
            except Exception as e:
                st.error(str(e))
