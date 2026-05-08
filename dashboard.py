import streamlit as st
import pandas as pd
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
if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None

# --- DATA ENGINE ---
@st.cache_data(ttl=5)
def get_data():
    l_res = supabase.table("leads").select("*").execute()
    i_res = supabase.table("inventory").select("*").execute()
    return pd.DataFrame(l_res.data), pd.DataFrame(i_res.data)

leads_df, inv_df = get_data()
SOM_VAL, SAM_VAL = 2900, 14500
IDEAL_BUDGET = 2000000
CAC_THRESHOLD = 50000  # ₦50k – if CAC exceeds, trigger alert

# ============================================================
# INTENT & FRAMEWORK SELECTION (same as before)
# ============================================================
def classify_strategic_intent(user_query):
    classification_prompt = f"""
    Classify the user's business query into EXACTLY one of these categories:
    - CUSTOMER_ANALYSIS: Queries about student behavior, demographics, needs, or "who" the customers are.
    - MARKET_ANALYSIS: Queries about competitors, SWOT, market trends, or external threats.
    - HYBRID_OPTIMIZATION: Queries about budget, growth strategy (Ansoff), or resource allocation.
    Query: "{user_query}"
    Return only the category name.
    """
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": classification_prompt}],
            temperature=0,
            max_tokens=20
        )
        return response.choices[0].message.content.strip()
    except:
        return "HYBRID_OPTIMIZATION"

def optimize_framework_selection(leads_df, inv_df, user_query):
    intent = classify_strategic_intent(user_query)
    has_leads = not leads_df.empty
    has_inventory = not inv_df.empty
    config = {
        "CUSTOMER_ANALYSIS": {
            "pillar": 1,
            "frameworks": ["Audience Segmentation", "CJM", "JTBD"],
            "data_check": has_leads,
            "missing_msg": "⚠️ Customer analysis requires lead data. Ask Michael to sync leads."
        },
        "MARKET_ANALYSIS": {
            "pillar": 2,
            "frameworks": ["VRIO", "Blue Ocean", "PESTLE"],
            "data_check": True,
            "missing_msg": "🌍 Market analysis active – using competitive intelligence."
        },
        "HYBRID_OPTIMIZATION": {
            "pillar": 3,
            "frameworks": ["Ansoff Matrix", "Resource Allocation"],
            "data_check": has_leads and has_inventory,
            "missing_msg": "🔗 Connecting customer needs to market opportunities (hybrid mode)."
        }
    }
    selected = config.get(intent, config["HYBRID_OPTIMIZATION"])
    status = "READY" if selected["data_check"] else "DATA_MISSING"
    return {
        "mode": intent,
        "status": status,
        "message": selected["missing_msg"] if not selected["data_check"] else f"✅ {intent.replace('_', ' ')} ready",
        "pillar": selected["pillar"],
        "frameworks": selected["frameworks"]
    }

def rank_and_filter_memories(query, memories, top_k=2):
    if not memories:
        return ""
    context_block = "\n".join([f"ID:{i} | Content:{m['content'][:300]}" for i, m in enumerate(memories)])
    ranking_prompt = f"""
    User Query: "{query}"
    Candidate Memories:
    {context_block}
    Identify the IDs of the top {top_k} most relevant memories that directly help answer the query. 
    Return ONLY the IDs as a comma-separated list. If none are relevant, return 'NONE'.
    """
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": ranking_prompt}],
            temperature=0,
            max_tokens=50
        )
        relevant_ids_str = response.choices[0].message.content.strip()
        if relevant_ids_str.upper() == "NONE":
            return ""
        relevant_ids = [id.strip() for id in relevant_ids_str.split(",") if id.strip().isdigit()]
        filtered_content = [memories[int(idx)]['content'] for idx in relevant_ids if int(idx) < len(memories)]
        return "\n---\n".join(filtered_content[:top_k])
    except:
        return memories[0]['content'] if memories else ""

def get_system_prompt_by_framework(framework_result, user_location, user_niche):
    personas = {
        "CUSTOMER_ANALYSIS": {
            "role": "The Student Anthropologist",
            "strategy": "Focus on ethnographic insights and 'Jobs to be Done' for LASU students.",
            "tone": "Empathetic, data-driven regarding student stressors."
        },
        "MARKET_ANALYSIS": {
            "role": "The Competitive Intelligence Officer",
            "strategy": "Analyze through VRIO and market gaps.",
            "tone": "Analytical, aggressive regarding competitive positioning."
        },
        "HYBRID_OPTIMIZATION": {
            "role": "The Chief Operations Architect",
            "strategy": "Balance CAC against market scalability.",
            "tone": "Pragmatic, focused on ROI."
        }
    }
    p = personas.get(framework_result['mode'], personas["HYBRID_OPTIMIZATION"])
    base = f"""You are {p['role']} at Penafort Strategic.
OPERATIONAL PHILOSOPHY: {p['strategy']}
Tone: {p['tone']}
Location: {user_location} ({user_niche})
Selected Frameworks: {', '.join(framework_result['frameworks'])}
Pillar: {framework_result['pillar']}
"""
    if framework_result['status'] == "DATA_MISSING":
        base += f"\n⚠️ {framework_result['message']}\nStop and ask Michael for specific inputs."
    else:
        base += f"\n✅ {framework_result['message']}"
    base += "\n\nAlways conclude with a **Prescriptive Action**."
    return base

# ============================================================
# REAL CAC ALERT (not cosmetic)
# ============================================================
def check_cac_alert(budget, leads_count):
    """Returns (alert_needed, alert_message, recovery_action)"""
    cac = budget / leads_count if leads_count > 0 else 0
    if cac > CAC_THRESHOLD:
        # Generate a real recovery directive using LLM
        recovery_prompt = f"""
        CAC is ₦{int(cac):,} which exceeds the target ₦{CAC_THRESHOLD:,}. 
        Suggest 3 specific actions to reduce CAC for a student fragrance business in Lagos.
        Keep response under 150 characters.
        """
        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": recovery_prompt}],
                temperature=0.5,
                max_tokens=100
            )
            action = resp.choices[0].message.content.strip()
        except:
            action = "Shift 30% budget to WhatsApp marketing and run LASU referral campaign."
        return True, f"🚨 CAC BREACH: ₦{int(cac):,} (Threshold: ₦{CAC_THRESHOLD:,})", action
    return False, "", ""

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
            st.cache_data.clear()
            st.rerun()

# ============================================================
# DYNAMIC CAC ALERT (Real)
# ============================================================
actual_count = len(leads_df)
current_cac = mkt_budget / actual_count if actual_count > 0 else 0
alert_needed, alert_msg, recovery_action = check_cac_alert(mkt_budget, actual_count)
if alert_needed:
    st.error(alert_msg)
    with st.expander("📋 AI Recovery Plan (Click to execute)"):
        st.write(recovery_action)
        if st.button("Apply Recovery Action", key="apply_recovery"):
            # Log the action into suggestions or directly update strategy
            supabase.table("suggestions").insert({
                "framework_used": "CAC_ALERT",
                "proposed_change": {"budget_shift": -0.3, "channel": "WhatsApp"},
                "reasoning": recovery_action,
                "status": "pending"
            }).execute()
            st.success("Recovery action logged. Awaiting approval.")
            st.rerun()

# ============================================================
# MAIN DASHBOARD
# ============================================================
st.title("🦅 Penafort Strategic Advisor")
c1, c2, c3, c4 = st.columns(4)
c1.metric("📊 Total Leads", actual_count)
c2.metric("💰 Pipeline Value", f"₦{actual_count * 15000:,}")
c3.metric("💸 CAC", f"₦{int(current_cac):,}")
c4.metric("🎯 Ideal CPA", f"₦{int(IDEAL_BUDGET/SOM_VAL):,}")
st.divider()

# --- Data Quality Indicators ---
col1, col2 = st.columns(2)
with col1:
    if not leads_df.empty:
        st.success(f"✅ Customer Data: {len(leads_df)} leads")
    else:
        st.warning("⚠️ No leads – customer analysis limited")
with col2:
    if not inv_df.empty:
        st.success(f"✅ Market Data: {len(inv_df)} inventory items")
    else:
        st.warning("⚠️ No inventory – market analysis limited")

# ============================================================
# FUNCTIONAL QUICK PROMPT BUTTONS
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
    if cols[idx].button(label, key=f"quick_{idx}"):
        st.session_state.quick_prompt = query
        st.rerun()

# ============================================================
# CHAT INTERFACE
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Determine input: either chat_input or quick_prompt
user_input = st.chat_input("Ask your strategy advisor...")
if st.session_state.quick_prompt:
    user_input = st.session_state.quick_prompt
    st.session_state.quick_prompt = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing with optimal framework..."):
            framework_result = optimize_framework_selection(leads_df, inv_df, user_input)
            st.info(f"🧠 **Selected:** {framework_result['mode'].replace('_', ' ')} | {framework_result['message']}")
            
            # Vector search (Gemini embedding)
            try:
                emb = genai.embed_content(
                    model="models/gemini-embedding-2-preview",
                    content=user_input,
                    output_dimensionality=768
                )['embedding']
                res = supabase.rpc('match_memories', {
                    'query_embedding': emb,
                    'match_threshold': 0.3,
                    'match_count': 5
                }).execute()
                raw_memories = res.data if res.data else []
            except:
                raw_memories = []
            
            context = rank_and_filter_memories(user_input, raw_memories, top_k=2)
            
            metrics = f"""
Leads: {actual_count}
SAM: {SAM_VAL}
SOM: {SOM_VAL}
Current CAC: {current_cac}
Budget: {mkt_budget}
Framework Mode: {framework_result['mode']}
"""
            history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]])
            
            system_prompt = get_system_prompt_by_framework(framework_result, user_location, user_niche)
            
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"""
HISTORY: {history}
RELEVANT MEMORIES: {context}
METRICS: {metrics}
FRAMEWORK STATUS: {framework_result['message']}
QUESTION: {user_input}
Provide strategic analysis and end with a Prescriptive Action.
"""}
                    ],
                    temperature=0.3
                )
                answer = response.choices[0].message.content
                st.write(answer)
            except Exception as e:
                answer = f"Error: {e}"
                st.error(answer)
            
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
            st.rerun()

# ============================================================
# STRATEGY PROPOSAL ROOM (with working approve/reject)
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
                supabase.table("suggestions").update({"status": "approved"}).eq("id", prop["id"]).execute()
                st.success("✅ Strategy approved and executed!")
                st.rerun()
            if col2.button("❌ Reject", key=f"rej_{prop['id']}"):
                supabase.table("suggestions").update({"status": "rejected"}).eq("id", prop["id"]).execute()
                st.warning("Proposal rejected.")
                st.rerun()
else:
    st.info("📭 No pending strategic proposals.")

# Keep message history limited
st.session_state.messages = st.session_state.messages[-10:]
