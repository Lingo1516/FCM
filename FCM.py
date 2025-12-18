import streamlit as st
import requests

st.set_page_config(page_title="金鑰最終驗證", page_icon="🔑")

st.title("🔑 Google API 金鑰最終驗證")
st.info("請在此測試您從 Google AI Studio 申請的「新專案」金鑰。")

# 讓使用者輸入 Key
user_key = st.text_input("請貼上您的 API Key (AIza 開頭)：", type="password")

if st.button("🚀 立即驗證", type="primary"):
    if not user_key:
        st.warning("請先貼上金鑰！")
    else:
        with st.spinner("正在連線 Google 伺服器..."):
            # 測試指令
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={user_key}"
            headers = {'Content-Type': 'application/json'}
            data = {"contents": [{"parts": [{"text": "Hello"}]}]}
            
            try:
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    st.success("✅ 驗證成功！這把鑰匙是有效的！")
                    st.json(response.json()) # 秀出 Google 回傳的證據
                    st.balloons()
                    st.markdown("### 🎉 恭喜！現在你可以把這把鑰匙拿去跑分析程式了！")
                elif response.status_code == 404:
                    st.error("❌ 驗證失敗 (404)")
                    st.error("原因：這把鑰匙沒有權限。請確定您是在 **Google AI Studio** 按下 **Create in new project** 申請的。")
                elif response.status_code == 429:
                    st.error("❌ 驗證失敗 (429)")
                    st.error("原因：額度已滿。請稍等幾分鐘或申請新專案。")
                else:
                    st.error(f"❌ 驗證失敗 (代碼 {response.status_code})")
                    st.text(response.text)
            except Exception as e:
                st.error(f"連線錯誤：{e}")
