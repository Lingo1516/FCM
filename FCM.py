# --- Tab 3: AI 策略顧問 (邏輯修正與增強版) ---
with tab3:
    st.subheader("🤖 論文深度分析顧問")
    
    # 對話視窗容器
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">👤 <b>您：</b>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-ai">🤖 <b>AI：</b>{msg["content"]}</div>', unsafe_allow_html=True)

    user_input = st.text_input("請輸入您的問題 (例如：請詳細分析每一個準則的變化)", key="chat_input")
    
    if st.button("送出問題") and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        if st.session_state.last_results is None:
            response = "⚠️ 請先至「模擬運算」跑出數據，我才能分析。"
        else:
            # 準備數據
            results = st.session_state.last_results # Shape: (Steps, Concepts)
            steps = results.shape[0]
            concepts = st.session_state.concepts
            initial = results[0]
            final = results[-1]
            growth = final - initial
            matrix = st.session_state.matrix
            
            response = ""

            # =================================================
            # 邏輯 1: 最高優先級 - 使用者想看「每一個」細節
            # =================================================
            if "每一" in user_input or "詳細" in user_input or "全部" in user_input:
                response += "### 📋 全方位準則深度解析報告\n\n"
                response += "本報告針對模型中的所有準則，分析其從策略介入到收斂的完整動態：\n\n"
                
                for i, c in enumerate(concepts):
                    # 1. 數據特徵
                    init_v = initial[i]
                    final_v = final[i]
                    grow_v = growth[i]
                    
                    # 2. 判斷角色 (Driver / Receiver)
                    role_str = ""
                    if init_v > 0.1:
                        role_str = "🔴 主動策略 (Driver)"
                    elif grow_v > 0.1:
                        role_str = "🟢 關鍵受惠者 (Receiver)"
                    elif final_v < 0.05:
                        role_str = "⚪ 沉寂指標 (Inactive)"
                    else:
                        role_str = "🔵 一般連動指標"

                    # 3. 找原因 (誰影響了它？)
                    # 檢查矩陣的 Column，看誰給它正權重
                    incoming_weights = matrix[:, i]
                    drivers = []
                    for src_idx, w in enumerate(incoming_weights):
                        if w > 0.1: drivers.append(f"{concepts[src_idx]}(權重{w})")
                    driver_text = "、".join(drivers) if drivers else "無顯著外部驅動力"

                    # 4. 寫入段落
                    response += f"#### **{c}** {role_str}\n"
                    response += f"- **【數值變化】**：初始 {init_v:.2f} $\\rightarrow$ 最終 {final_v:.2f} (成長幅度 {grow_v:+.2f})\n"
                    response += f"- **【驅動來源】**：其數值變化主要受 **[{driver_text}]** 的影響。\n"
                    
                    # 5. 回合/階段分析 (如果有變化的話)
                    if grow_v > 0.01:
                        # 取出早中晚三個時間點
                        mid_step = int(steps / 2)
                        early_val = results[min(5, steps-1), i]
                        mid_val = results[mid_step, i]
                        
                        response += f"- **【時序階段】**：\n"
                        response += f"  - *初期 (Step 1-5)*：數值由 {init_v:.2f} 爬升至 {early_val:.2f} (啟動期)。\n"
                        response += f"  - *中期 (Step {mid_step})*：加速成長至 {mid_val:.2f} (擴散期)。\n"
                        response += f"  - *後期 (Step {steps})*：收斂穩定於 {final_v:.2f} (穩定期)。\n"
                    
                    response += "\n---\n"

            # =================================================
            # 邏輯 2: 使用者問「回合」或「過程」
            # =================================================
            elif "回合" in user_input or "過程" in user_input or "時間" in user_input:
                response += "### ⏳ 系統動態時序分析 (Time-Series Analysis)\n\n"
                response += "FCM 的模擬過程可分為三個關鍵階段，這對於解釋策略的「時間滯後性 (Time Lag)」非常有幫助：\n\n"
                
                # 找出變動最大的前 3 名來舉例
                top_growers = np.argsort(growth)[::-1][:3]
                
                response += "**第一階段：策略震盪期 (Step 0-10)**\n"
                response += "在此階段，策略剛剛介入。您會觀察到直接投入的因子 (Driver) 數值瞬間拉高，但下游因子尚未反應。這在管理上對應於「組織內部的適應與磨合期」。\n\n"
                
                response += "**第二階段：連鎖擴散期 (Step 10-25)**\n"
                response += "這是系統變化最劇烈的時期。矩陣中的因果路徑開始發酵。數據顯示，"
                for idx in top_growers:
                    if growth[idx] > 0.05:
                        response += f"**{concepts[idx]}** 開始顯著爬升、"
                response += "顯示跨部門的綜效正在產生。\n\n"
                
                response += f"**第三階段：動態穩定期 (Step {steps})**\n"
                response += "系統各項數值不再變動，達到「收斂 (Convergence)」。這代表組織已形成新的文化與運作慣性 (Routine)。\n"

            # =================================================
            # 邏輯 3: 一般解釋 (優先級最低)
            # =================================================
            else:
                best_idx = np.argmax(final)
                driver_idx = np.argmax(initial)
                response += f"根據模擬，**{concepts[best_idx]}** 表現最佳。\n"
                response += f"若您需要詳細報告，請輸入「解釋每一個準則」或「分析每一個回合」。"

        st.session_state.chat_history.append({"role": "ai", "content": response})
        st.rerun()
