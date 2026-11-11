"""Streamlit chat UI.  Run: streamlit run ui/app.py  (API must be running on API_URL)."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="সেবা সহায়ক · Sheba RAG", page_icon=":material/account_balance:")
st.title("সেবা সহায়ক")
st.caption("পাসপোর্ট, এনআইডি ও ভূমি সেবা নিয়ে বাংলা বা ইংরেজিতে প্রশ্ন করুন। "
           "Answers come only from official documents, with sources.")

with st.sidebar:
    st.subheader("উদাহরণ প্রশ্ন")
    examples = ["হারানো এনআইডি কার্ড আবার পেতে কী করতে হবে?", "নামজারি করতে কত টাকা ফি লাগে?",
                "অতি জরুরী পাসপোর্ট কত দিনে পাওয়া যায়?", "What is the NID helpline number?"]
    clicked = [st.button(q, width="stretch", key=f"example_{i}") for i, q in enumerate(examples)]  # render all buttons every run
    picked = next((q for q, c in zip(examples, clicked) if c), None)
    st.divider()
    st.caption("This is a portfolio demo, not legal advice. Always confirm with the official office.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for c in msg.get("citations", []):
            where = ", ".join(x for x in (f"§ {c['section']}" if c.get("section") else "",
                                          f"p. {c['page']}" if c.get("page") else "") if x)
            st.caption(f"[{c['n']}] [{c['title']}]({c['url']})" + (f" — {where}" if where else ""))


for m in st.session_state.messages:
    render(m)

question = (st.chat_input("আপনার প্রশ্ন লিখুন…", submit_mode="disable", key="question") or picked or "").strip()
if len(question) == 1:  # the API rejects anything under 2 characters; say so plainly instead of showing its error
    st.warning("অনুগ্রহ করে একটি প্রশ্ন লিখুন। Please type a question.")
elif question:
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    render(user_msg)
    with st.spinner("উত্তর খোঁজা হচ্ছে…"):
        try:
            r = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=120)
            if r.ok:
                data = r.json()
            else:
                detail = r.json().get("detail", r.text) if "json" in r.headers.get("content-type", "") else r.text
                data = {"answer": f"⚠️ {detail}", "citations": []}
        except requests.Timeout:  # the API can be busy retrying a slow LLM; it is not down
            data = {"answer": "⚠️ The answer is taking too long. Please try again.", "citations": []}
        except requests.RequestException as e:
            data = {"answer": f"⚠️ API not reachable at {API_URL}: {e}", "citations": []}
    bot_msg = {"role": "assistant", "content": data["answer"], "citations": data.get("citations", [])}
    st.session_state.messages.append(bot_msg)
    render(bot_msg)
