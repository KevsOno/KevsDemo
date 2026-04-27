import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from supabase import create_client
import google.generativeai as genai
from groq import Groq
from datetime import datetime
import logging
import numpy as np

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

# ============================================================
# INTELLIGENT FRAMEWORK OPTIMIZER
# ============================================================

def optimize_framework_selection(leads_df, inv_df, user_query, user_context=None):
    """
    Determines which mode (Customer, Market, or Hybrid) 
    has enough data to be accurate.
    """
    has_leads = not leads_df.empty
    has_inventory = not inv_df.empty
    
    # Customer-focused queries
    if "who" in user_query.lower() or "customer" in user_query.lower() or "student" in user_query.lower():
        if has_leads:
            return {
                "mode": "CUSTOMER_ANALYSIS",
                "status": "READY",
                "message": "✅ Customer analysis ready - lead data available",
                "pillar": 1,
                "frameworks": ["Audience Segmentation", "CJM", "Value Prop Canvas", "JTBD", "NPS", "PMF"]
            }
        return {
            "mode": "CUSTOMER_ANALYSIS",
            "status": "DATA_MISSING",
            "message": "⚠️ Customer analysis requires lead data. Ask Michael to sync leads first.",
            "pillar": 1,
            "frameworks": ["Audience Segmentation", "CJM", "Value Prop Canvas", "JTBD", "NPS", "PMF"]
        }
    
    # Market-focused queries
    if "competitor" in user_query.lower() or "market" in user_query.lower() or "swot" in user_query.lower():
        return {
            "mode": "MARKET_ANALYSIS",
            "status": "REQUIRES_EXTERNAL",
            "message": "🌍 Market analysis active - using competitive intelligence frameworks",
            "pillar": 2,
            "frameworks": ["SWOT", "VRIO", "Value Chain", "BCG Matrix", "Ansoff", "Blue Ocean", "PESTLE"]
        }
    
    # Default to HYBRID with data check
    data_quality = "HIGH" if has_leads and has_inventory else "MEDIUM" if (has_leads or has_inventory) else "LOW"
    
    return {
        "mode": "HYBRID_OPTIMIZATION",
        "status": "READY",
        "data_quality": data_quality,
        "message": f"🔗 Hybrid optimization active (Data Quality: {data_quality}) - connecting customer needs to market opportunities",
        "pillar": 3,
        "frameworks": ["Cross-functional optimization", "Resource allocation", "Strategic synthesis"]
    }

def get_system_prompt_by_framework(framework_result, user_location=None, user_niche=None):
    """
    Generates dynamic system prompt based on selected framework and context
    """
    location_context = user_location if user_location else "Lagos/LASU"
    niche_context = user_niche if user_niche else "student fragrance market"
    
    base_prompt = f"""You are the Penafort Strategy Engine, a specialist in Prescriptive Analytics and Operations Research.

### PILLAR {framework_result['pillar']}: {framework_result['mode'].replace('_', ' ').title()}
- **Selected Frameworks:** {', '.join(framework_result['frameworks'][:3])}
- **Goal:** {get_pillar_goal(framework_result['pillar'])}

### CORE OPERATING RULES:
1. **MARKET PERSPECTIVE:** Use phrases like "The market indicates..." or "Compared to competitors..."
2. **DATA INTEGRITY:** If specific data for a framework is missing, ask: "To provide deeper insight, could you share [specific data needed]?"
3. **DYNAMIC LOCALIZATION:** Adapt your analysis to {location_context} - specifically the {niche_context}.
4. **OPTIMIZATION:** Always select the framework that requires the least assumptions based on currently available data.
"""
    
    if framework_result['status'] == 'DATA_MISSING':
        base_prompt += f"\n\n⚠️ **DATA WARNING:** {framework_result['message']}\nGuide Michael on what data to collect before running this analysis."
    
    return base_prompt

def get_pillar_goal(pillar):
    if pillar == 1:
        return "Aligning product-market fit through deep customer understanding"
    elif pillar == 2:
        return "Exploiting competitive advantages and identifying market gaps"
    else:
        return "Cross-functional optimization and strategic resource allocation"

# ============================================================
# TOOL FUNCTIONS
# ============================================================

def propose_strategy_update(framework: str, suggestion: str, new_values: dict):
    """Propose changes with Lagos context"""
    local_note = "Optimized for Lagos delivery routes and LASU student peak hours."
    enhanced_suggestion = f"{suggestion}\n\n{local_note}"
    
    try:
        supabase.table("suggestions").insert({
            "framework_used": framework,
            "proposed_change": new_values,
            "reasoning": enhanced_suggestion
        }).execute()
        return f"✅ Strategic proposal logged via {framework}. Awaiting Michael's approval."
    except Exception as e:
        return f"❌ Error logging proposal: {str(e)}"

def run_optimization_analysis(budget: float, lead_data: list):
    """Calculate best budget allocation"""
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
        "recommendation": f"🎯 Optimization suggests shifting 40% of budget to {best_faculty}"
    }

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("🎮 Management Controls")
    mkt_budget = st.slider("Budget (₦)", 50000, 2000000, 500000, 50000)
    
    st.header("📍 Context Settings")
    user_location = st.text_input("Target Location", value="Lagos, LASU")
    user_niche = st.text_input("Target Niche", value="Student fragrance market")
    
    st.header("📍 Field Entry")
    with st.form("registration_form", clear_on_submit=True):
        name = st.text_input("Student Name")
        fac = st.selectbox("Faculty", ["Arts", "Science", "Law", "Engineering", "Management", "Social Sciences"])
        if st.form_submit_button("Sync Lead") and name:
            supabase.table("leads").insert({"student_name": name, "faculty": fac}).execute()
            st.success(f"✅ Lead {name} Synced!")
            st.rerun()

# ============================================================
# ALERTS
# ============================================================
alert_res = supabase.table("market_alerts").select("*").order("created_at", desc=True).limit(1).execute()
if alert_res.data:
    latest = alert_res.data[0]
    if latest['status'] == 'breach':
        st.error(f"🚨 CAC BREACH DETECTED (₦{int(latest['metric_value']):,})")
        with st.expander("📋 AI Recovery Plan"):
            st.write(latest['ai_directive'])

# ============================================================
# MAIN DASHBOARD
# ============================================================
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

# ============================================================
# DYNAMIC FRAMEWORK STATUS
# ============================================================
st.subheader("🧠 Active Strategy Engine")

# Show current data availability
col1, col2 = st.columns(2)
with col1:
    if not leads_df.empty:
        st.success(f"✅ Customer Data: {len(leads_df)} leads available")
    else:
        st.warning("⚠️ Customer Data: No leads - customer analysis limited")
with col2:
    if not inv_df.empty:
        st.success(f"✅ Market Data: {len(inv_df)} inventory items")
    else:
        st.warning("⚠️ Market Data: No inventory - market analysis limited")

# ============================================================
# QUICK PROMPTS WITH INTELLIGENT ROUTING
# ============================================================
st.subheader("💡 Strategic Questions")

prompts = {
    "👥 Who are our customers?": "Who are our primary customers and what do they need?",
    "🏪 How do we beat competitors?": "Analyze our competitive position using VRIO framework",
    "🚀 Where should we grow?": "What growth strategy should we pursue using Ansoff Matrix?",
    "🔗 Integrated view": "Connect our customer needs to market opportunities"
}

cols = st.columns(4)
for idx, (label, query) in enumerate(prompts.items()):
    if cols[idx].button(label):
        st.session_state.messages.append({"role": "user", "content": query})

st.divider()

# ============================================================
# CHAT INTERFACE WITH INTELLIGENT FRAMEWORK SELECTION
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask your strategy advisor...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Selecting optimal framework..."):
            
            # Step 1: Intelligently select framework based on query + available data
            framework_result = optimize_framework_selection(leads_df, inv_df, user_input)
            
            # Show which framework was selected
            st.info(f"🧠 **Selected:** {framework_result['mode'].replace('_', ' ')} | {framework_result['message']}")
            
            # Step 2: Embed query for vector search
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
Framework Mode: {framework_result['mode']}
Data Quality: {framework_result.get('data_quality', 'UNKNOWN')}
"""
            
            # History
            history = "\n".join([
                f"{m['role']}: {m['content']}"
                for m in st.session_state.messages[-6:]
            ])
            
            # Get dynamic system prompt
            system_prompt = get_system_prompt_by_framework(
                framework_result, 
                user_location=user_location,
                user_niche=user_niche
            )
            
            # AI response with framework context
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""
CONVERSATION:
{history}

RETRIEVED CONTEXT:
{context}

BUSINESS DATA:
{metrics}

QUESTION:
{user_input}

FRAMEWORK STATUS:
{framework_result['message']}

Please provide your analysis using the selected framework.
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
                    "framework_used": framework_result['mode'],
                    "timestamp": datetime.now().isoformat()
                }).execute()
            except:
                pass

# ============================================================
# STRATEGY PROPOSAL ROOM
# ============================================================
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

# --- LIMIT MEMORY ---
st.session_state.messages = st.session_state.messages[-10:]
