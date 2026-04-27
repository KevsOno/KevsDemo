import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from supabase import create_client
import google.generativeai as genai
from groq import Groq
from datetime import datetime
import logging

# --- PAGE CONFIG ---
st.set_page_config(page_title="Penafort Strategic Advisor", layout="wide")

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SECRETS ---
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# --- CLIENTS ---
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- DATA ENGINE ---
@st.cache_data(ttl=5)
def get_data():
    l_res = supabase.table("leads").select("*").execute()
    i_res = supabase.table("inventory").select("*").execute()
    return pd.DataFrame(l_res.data), pd.DataFrame(i_res.data)

leads_df, inv_df = get_data()
SOM_VAL, SAM_VAL = 2900, 14500
IDEAL_BUDGET = 2000000

# --- TOOL FUNCTIONS ---
def propose_strategy_update(framework: str, suggestion: str, new_values: dict):
    """Propose changes to business metrics or strategy."""
    try:
        supabase.table("suggestions").insert({
            "framework_used": framework,
            "proposed_change": new_values,
            "reasoning": suggestion
        }).execute()
        return f"✅ Strategic proposal logged via {framework}. Awaiting approval."
    except Exception as e:
        return f"❌ Error logging proposal: {str(e)}"

def run_optimization_analysis(budget: float, lead_data: list):
    """Calculate best budget allocation across faculties."""
    import pandas as pd
    df = pd.DataFrame(lead_data)
    
    if df.empty:
        return {"error": "No lead data available"}
    
    summary = df.groupby('faculty').size().reset_index(name='count')
    summary['efficiency_score'] = summary['count'] / summary['count'].sum()
    summary['suggested_spend'] = summary['efficiency_score'] * budget
    
    best_faculty = summary.sort_values(by='efficiency_score', ascending=False).iloc[0]['faculty']
    
    return {
        "best_move": best_faculty,
        "allocation_plan": summary.to_dict(orient='records'),
        "recommendation": f"🎯 Optimization suggests shifting 40% of budget to {best_faculty} to maximize SOM growth."
    }

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
            st.success(f"✅ Lead {name} Synced!")
            st.rerun()

# --- ALERTS ---
alert_res = supabase.table("market_alerts").select("*").order("created_at", desc=True).limit(1).execute()
if alert_res.data:
    latest = alert_res.data[0]
    if latest['status'] == 'breach':
        st.error(f"🚨 CAC BREACH DETECTED (₦{int(latest['metric_value']):,})")
        with st.expander("📋 AI Recovery Plan"):
            st.write(latest['ai_directive'])

# --- MAIN DASHBOARD ---
st.title("🦅 Penafort Strategic Advisor")

# --- KPIs ---
actual_count = len(leads_df)
current_cac = mkt_budget / actual_count if actual_count > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("📊 Total Leads", actual_count)
c2.metric("💰 Pipeline Value", f"₦{actual_count * 15000:,}")
c3.metric("💸 CAC", f"₦{int(current_cac):,}")
c4.metric("🎯 Ideal CPA", f"₦{int(IDEAL_BUDGET/SOM_VAL):,}")

st.divider()

# --- QUICK PROMPTS ---
st.subheader("💡 Quick Questions")
col1, col2, col3 = st.columns(3)

if col1.button("📍 Where should we scale?"):
    st.session_state.messages.append({"role": "user", "content": "Where should we scale?"})

if col2.button("⚠️ Are we overspending?"):
    st.session_state.messages.append({"role": "user", "content": "Analyze our CAC efficiency"})

if col3.button("🚀 How do we hit SOM faster?"):
    st.session_state.messages.append({"role": "user", "content": "How can we reach our SOM target faster?"})

# --- CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- CHAT INPUT ---
user_input = st.chat_input("Ask your AI advisor...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Embed query
            emb = genai.embed_content(
                model="models/gemini-embedding-2-preview",
                content=user_input,
                output_dimensionality=768
            )['embedding']
            
            # Retrieve memory
            res = supabase.rpc('match_memories', {
                'query_embedding': emb,
                'match_threshold': 0.3,
                'match_count': 3
            }).execute()
            
            context = "\n".join([c['content'] for c in res.data]) if res.data else ""
            
            # Business metrics
            metrics = f"""
Leads: {actual_count}
SAM: {SAM_VAL}
SOM: {SOM_VAL}
Current CAC: {current_cac}
Budget: {mkt_budget}
"""
            
            # History
            history = "\n".join([
                f"{m['role']}: {m['content']}"
                for m in st.session_state.messages[-6:]
            ])
            
            # System prompt with tool awareness
            system_content = """You are the Penafort Lead Strategist.
            
You have access to these frameworks:
- VRIO: Analyze if resources are Valuable, Rare, Inimitable, Organized
- BCG Matrix: Categorize products (Stars, Cash Cows, Question Marks, Dogs)
- Value Proposition: Map customer pains to solutions

If you notice inefficient SAM/SOM or budget allocation, explain the issue clearly.
For strategic proposals, ask Michael to use the Strategy Proposal Room below."""
            
            # AI response
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"""
CONVERSATION:
{history}

RETRIEVED CONTEXT:
{context}

BUSINESS DATA:
{metrics}

QUESTION:
{user_input}
"""}
                ],
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Log to Supabase
            try:
                supabase.table("api_logs").insert({
                    "query": user_input[:500],
                    "response": answer[:500],
                    "budget": mkt_budget,
                    "leads_count": actual_count,
                    "timestamp": datetime.now().isoformat()
                }).execute()
            except:
                pass

# --- STRATEGY PROPOSAL ROOM ---
st.divider()
st.header("🧠 Strategy Proposal Room")

prop_res = supabase.table("suggestions").select("*").eq("status", "pending").execute()

if prop_res.data:
    for prop in prop_res.data:
        with st.expander(f"💡 Proposal: {prop['framework_used']}"):
            st.write(f"**Reasoning:** {prop['reasoning']}")
            st.json(prop['proposed_change'])
            
            col1, col2 = st.columns(2)
            if col1.button("✅ Approve & Execute", key=f"app_{prop['id']}"):
                st.success("✅ Strategy Integrated!")
                supabase.table("suggestions").update({"status": "approved"}).eq("id", prop["id"]).execute()
                st.rerun()
                
            if col2.button("❌ Reject", key=f"rej_{prop['id']}"):
                supabase.table("suggestions").update({"status": "rejected"}).eq("id", prop["id"]).execute()
                st.rerun()
else:
    st.info("📭 No pending strategic proposals.")

# --- VISUALIZATIONS ---
st.divider()
st.subheader("📊 Analytics")

if not leads_df.empty:
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(leads_df['faculty'].value_counts())
    with col2:
        st.dataframe(inv_df, use_container_width=True)

# --- LIMIT MEMORY ---
st.session_state.messages = st.session_state.messages[-10:]
