import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from supabase import create_client
import google.generativeai as genai
from groq import Groq

st.set_page_config(page_title="Penafort Strategic Advisor", layout="wide")

# --- SECRETS ---
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# --- CLIENTS ---
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# --- DATA ENGINE ---
@st.cache_data(ttl=5)
def get_data():
    l_res = supabase.table("leads").select("*").execute()
    i_res = supabase.table("inventory").select("*").execute()
    return pd.DataFrame(l_res.data), pd.DataFrame(i_res.data)

leads_df, inv_df = get_data()
SOM_VAL, SAM_VAL = 2900, 14500
IDEAL_BUDGET = 2000000

# --- SIDEBAR ---
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

# --- ALERT ---
alert_res = supabase.table("market_alerts").select("*").order("created_at", desc=True).limit(1).execute()
if alert_res.data:
    latest = alert_res.data[0]
    if latest['status'] == 'breach':
        st.error(f"🚨 CAC BREACH DETECTED (₦{int(latest['metric_value']):,})")
        with st.expander("AI Recovery Plan"):
            st.write(latest['ai_directive'])

st.title("🦅 Penafort Strategic Advisor")

# --- KPIs ---
actual_count = len(leads_df)
current_cac = mkt_budget / actual_count if actual_count > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Leads", actual_count)
c2.metric("Pipeline Value", f"₦{actual_count * 15000:,}")
c3.metric("CAC", f"₦{int(current_cac):,}")
c4.metric("Ideal CPA", f"₦{int(IDEAL_BUDGET/SOM_VAL):,}")

st.divider()

# --- AI SEARCH ---
query = st.text_input("Ask Strategy Vault...")
if query:
    emb = genai.embed_content(
        model="models/gemini-embedding-2-preview",
        content=query,
        output_dimensionality=768
    )['embedding']

    res = supabase.rpc('match_memories', {
        'query_embedding': emb,
        'match_threshold': 0.3,
        'match_count': 2
    }).execute()

    if res.data:
        context = "\n".join([c['content'] for c in res.data])
        ans = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Senior strategist."},
                {"role": "user", "content": f"{context}\n{query}"}
            ]
        ).choices[0].message.content

        st.write(ans)

# --- VISUALS ---
if not leads_df.empty:
    st.bar_chart(mkt_budget / leads_df['faculty'].value_counts())

st.dataframe(inv_df, use_container_width=True)
