import streamlit as st
import requests

st.set_page_config(page_title="真相診斷器", page_icon="🕵️")

st.title("🕵️ API Key 真相診斷")
st.info("我們不猜了，直接問 Google 這把鑰匙能看到什麼。")

# 1. 請貼上截圖中那把鑰匙
default_key = st.text_input("請貼上結尾是 WY0iw 的那把鑰匙：", value="")

if st.button("🚀 執行診斷", type="primary"):
    if len(default_key) < 10:
        st.warning("請先貼上完整的鑰匙！")
    else:
        # 2. 直接向 Google 請求「模型清單」 (ListModels)
        # 這是最底層的查詢，如果這個都失敗，代表專案真的壞了
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={default_key}"
        
        try:
            response = requests.get(url)
            
            if response.status_code == 200:
                # --- 情況 A：成功 (代表鑰匙是好的) ---
                data = response.json()
                models = [m['name'] for m in data.get('models', [])]
                
                st.success(f"✅ 診斷成功！您的鑰匙有效，可以看到 {len(models)} 個模型。")
                st.write("Google 說您可以用這些模型：")
                st.json(models)
                
                # 自動幫你寫好這把鑰匙的分析程式
                st.divider()
                st.subheader("🎉 既然鑰匙是好的，請用這個區塊開始分析：")
                text_input = st.text_area("輸入文獻：")
                if st.button("開始分析"):
                    # 使用清單中的第一個 gemini 模型
                    valid_model = next((m for m in models if 'gemini' in m), 'models/gemini-pro')
                    gen_url = f"https://generativelanguage.googleapis.com/v1beta/{valid_model}:generateContent?key={default_key}"
                    r = requests.post(gen_url, json={"contents": [{"parts": [{"text": f"抓重點:{text_input}"}]}]})
                    st.write(r.json())
                    
            elif response.status_code == 404:
                # --- 情況 B：404 (代表專案沒開通) ---
                st.error("❌ 診斷結果：404 Not Found")
                st.error(f"嚴重問題：您的鑰匙 `{default_key[-5:]}` 雖然存在，但所屬的專案 `770509881178` **沒有啟用 API 服務**。")
                st.warning("這就是為什麼不管怎麼試都失敗的原因。這個專案壞了。")
                
            else:
                st.error(f"❌ 其他錯誤：{response.status_code}")
                st.text(response.text)
                
        except Exception as e:
            st.error(f"連線錯誤：{e}")
