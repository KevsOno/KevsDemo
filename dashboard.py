import os
import urllib
from google.colab import userdata

# Environment Setup
os.environ["SUPABASE_URL"] = userdata.get('SUPABASE_URL').strip().rstrip('/')
os.environ["SUPABASE_KEY"] = userdata.get('SUPABASE_KEY')
os.environ["GEMINI_API_KEY"] = userdata.get('GEMINI_API_KEY')
os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')

dashboard_code = """
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from supabase import create_client
import google.generativeai as genai
from groq import Groq
import os

st.set_page_config(page_title="Penafort Strategic Advisor", layout="wide")

# Clients
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- DATA ENGINE ---
@st.cache_data(ttl=5)
def get_data():
    l_res = supabase.table("leads").select("*").execute()
    i_res = supabase.table("inventory").select("*").execute()
    return pd.DataFrame(l_res.data), pd.DataFrame(i_res.data)

leads_df, inv_df = get_data()
SOM_VAL, SAM_VAL = 2900, 14500
IDEAL_BUDGET = 2000000

# --- SIDEBAR: RESTORED REGISTRATION & CONTROLS ---
with st.sidebar:
    st.header("🎮 Management Controls")
    mkt_budget = st.slider("Budget (₦)", 50000, 2000000, 500000, 50000)
    
    st.header("📍 Field Entry")
    with st.form("registration_form", clear_on_submit=True):
        name = st.text_input("Student Name")
        fac = st.selectbox("Faculty", ["Arts", "Science", "Law", "Engineering", "Management", "Social Sciences"])
        if st.form_submit_button("Sync Lead") and name:
            supabase.table("leads").insert({"student_name": name, "faculty": fac}).execute()
            st.success(f"Lead {name} Synced!")
            st.rerun()

# --- TOP LEVEL AGENTIC ALERT (NEW) ---
alert_res = supabase.table("market_alerts").select("*").order("created_at", desc=True).limit(1).execute()
if alert_res.data:
    latest = alert_res.data[0]
    if latest['status'] == 'breach':
        st.error(f"🚨 **SENTINEL ALERT: CAC BREACH DETECTED (₦{int(latest['metric_value']):,})**")
        with st.expander("📝 View AI Recovery Plan"):
            st.write(latest['ai_directive'])
            st.caption(f"Audit Time: {latest['created_at']}")

st.title("🦅 Penafort Strategic Advisor")

# --- KPI ROW ---
actual_count = len(leads_df)
current_cac = mkt_budget / actual_count if actual_count > 0 else 0
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Leads", actual_count, f"{round((actual_count/SOM_VAL)*100, 1)}% SOM")
c2.metric("Market Pipeline", f"₦{actual_count * 15000:,}")
c3.metric("Dynamic CAC", f"₦{int(current_cac):,}")
c4.metric("Ideal CPA", f"₦{int(IDEAL_BUDGET/SOM_VAL):,}")

st.divider()

# --- AI SEARCH INTERFACE (RESTORED) ---
col_ai, col_dir = st.columns([2, 1])
with col_ai:
    st.subheader("🧠 Strategic Memory Audit")
    query = st.text_input("Ask the Vault...")
    if query:
        with st.status("🔍 Searching Strategy Vault..."):
            emb = genai.embed_content(model="models/gemini-embedding-2-preview", content=query, output_dimensionality=768)['embedding']
            res = supabase.rpc('match_memories', {'query_embedding': emb, 'match_threshold': 0.3, 'match_count': 2}).execute()
            if res.data:
                context = "\\n".join([c['content'] for c in res.data])
                ans = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": "Senior Strategist response based on context."},
                              {"role": "user", "content": f"Context: {context}\\nQuery: {query}"}]
                ).choices[0].message.content
                st.write(ans)

with col_dir:
    st.subheader("🎯 Spend Directive")
    if not leads_df.empty:
        best_fac = leads_df['faculty'].value_counts().idxmax()
        st.success(f"Scale Opportunity: **{best_fac}**")

st.divider()

# --- VISUALS & INVENTORY ---
v1, v2 = st.columns(2)
with v1:
    st.subheader("📊 CAC by Faculty")
    if not leads_df.empty:
        f_counts = leads_df['faculty'].value_counts()
        st.bar_chart(mkt_budget / f_counts)

with v2:
    st.subheader("📦 Inventory Status")
    st.dataframe(inv_df[['item_name', 'stock_quantity', 'unit_price']], use_container_width=True)
"""

with open('dashboard.py', 'w') as f:
    f.write(dashboard_code)

print("PASSWORD:", urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip())
!fuser -k 8501/tcp
!streamlit run dashboard.py & npx localtunnel --port 8501