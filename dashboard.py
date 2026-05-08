import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from supabase import create_client
import google.generativeai as genai
from groq import Groq
from datetime import datetime
import logging
import json

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
# 1. INTELLIGENT INTENT CLASSIFIER (Replaces Keywords)
# ============================================================
def classify_strategic_intent(user_query):
    """Uses a fast LLM to classify intent into strategic pillars."""
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
    except Exception as e:
        logger.warning(f"Intent classification failed: {e}. Falling back to HYBRID.")
        return "HYBRID_OPTIMIZATION"

def optimize_framework_selection(leads_df, inv_df, user_query):
    """Determines mode based on intent classification and data availability."""
    intent = classify_strategic_intent(user_query)
    has_leads = not leads_df.empty
    has_inventory = not inv_df.empty
    
    config = {
        "CUSTOMER_ANALYSIS": {
            "pillar": 1,
            "frameworks": ["Audience Segmentation", "CJM", "JTBD"],
            "data_check": has_leads,
            "missing_msg": "⚠️ Customer analysis requires lead data. Ask Michael to sync leads first."
        },
        "MARKET_ANALYSIS": {
            "pillar": 2,
            "frameworks": ["VRIO", "Blue Ocean", "PESTLE"],
            "data_check": True,   # market analysis can work with external knowledge
            "missing_msg": "🌍 Market analysis active – using competitive intelligence frameworks."
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

# ============================================================
# 2. CONTEXTUAL RE-RANKING (Replaces Raw Vector Injection)
# ============================================================
def rank_and_filter_memories(query, memories, top_k=2):
    """Filters retrieved memories for actual relevance using an LLM evaluator."""
    if not memories:
        return ""
    
    # Build a numbered list for the LLM to reference
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
    except Exception as e:
        logger.warning(f"Re-ranking failed: {e}. Using top vector result.")
        return memories[0]['content'] if memories else ""

# ============================================================
# 3. SPECIALIZED SYSTEM PROMPTS (Per Pillar Persona)
# ============================================================
def get_system_prompt_by_framework(framework_result, user_location, user_niche):
    """Generates a deep-dive persona based on the strategic pillar."""
    
    personas = {
        "CUSTOMER_ANALYSIS": {
            "role": "The Student Anthropologist",
            "strategy": "Focus on ethnographic insights and the 'Jobs to be Done' for LASU students.",
            "tone": "Empathetic, data-driven regarding student stressors and desires."
        },
        "MARKET_ANALYSIS": {
            "role": "The Competitive Intelligence Officer",
            "strategy": "Analyze through the lens of VRIO (Value, Rarity, Imitability, Organization) and market gaps.",
            "tone": "Analytical, aggressive regarding competitive positioning."
        },
        "HYBRID_OPTIMIZATION": {
            "role": "The Chief Operations Architect",
            "strategy": "Balance customer acquisition cost (CAC) against market scalability.",
            "tone": "Pragmatic, focused on ROI and resource efficiency."
        }
    }
    
    mode = framework_result['mode']
    p = personas.get(mode, personas["HYBRID_OPTIMIZATION"])
    
    base_prompt = f"""You are {p['role']} at Penafort Strategic.

OPERATIONAL PHILOSOPHY:
- {p['strategy']}
- Tone: {p['tone']}
- Location Context: {user_location} (specifically the {user_niche}).

SELECTED FRAMEWORKS: {', '.join(framework_result['frameworks'])}
PILLAR: {framework_result['pillar']}
"""
    
    if framework_result['status'] == "DATA_MISSING":
        base_prompt += f"\n⚠️ DATA WARNING: {framework_result['message']}\nStop and ask Michael for specific inputs before analyzing."
    else:
        base_prompt += f"\n✅ {framework_result['message']}"
    
    base_prompt += "\n\nAlways conclude with a **Prescriptive Action** that Michael can execute immediately."
    return base_prompt

# ============================================================
# TOOL FUNCTIONS (unchanged from original)
# ============================================================
def propose_strategy_update(framework: str, suggestion: str, new_values: dict):
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
# SIDEBAR (unchanged)
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
# ALERTS (unchanged)
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
# QUICK PROMPTS (unchanged)
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
# CHAT INTERFACE WITH ALL IMPROVEMENTS
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
        with st.spinner("Analyzing with optimal framework..."):
            # Step 1: Intent-based framework selection
            framework_result = optimize_framework_selection(leads_df, inv_df, user_input)
            st.info(f"🧠 **Selected:** {framework_result['mode'].replace('_', ' ')} | {framework_result['message']}")
            
            # Step 2: Vector search (Gemini embedding – unchanged)
            try:
                emb = genai.embed_content(
                    model="models/gemini-embedding-2-preview",  # NOT swapped to 004
                    content=user_input,
                    output_dimensionality=768
                )['embedding']
                
                res = supabase.rpc('match_memories', {
                    'query_embedding': emb,
                    'match_threshold': 0.3,
                    'match_count': 5   # retrieve more then re-rank
                }).execute()
                raw_memories = res.data if res.data else []
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                raw_memories = []
            
            # Step 3: Re-rank memories for relevance
            context = rank_and_filter_memories(user_input, raw_memories, top_k=2)
            
            # Step 4: Prepare metrics
            metrics = f"""
Leads: {actual_count}
SAM: {SAM_VAL}
SOM: {SOM_VAL}
Current CAC: {current_cac}
Budget: {mkt_budget}
Framework Mode: {framework_result['mode']}
Data Quality: {framework_result.get('data_quality', 'UNKNOWN')}
"""
            history = "\n".join([
                f"{m['role']}: {m['content']}"
                for m in st.session_state.messages[-6:]
            ])
            
            # Step 5: Specialized system prompt
            system_prompt = get_system_prompt_by_framework(
                framework_result,
                user_location=user_location,
                user_niche=user_niche
            )
            
            # Step 6: Generate final answer
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"""
CONVERSATION HISTORY:
{history}

RELEVANT MEMORIES (after re-ranking):
{context}

BUSINESS METRICS:
{metrics}

FRAMEWORK STATUS:
{framework_result['message']}

USER QUESTION:
{user_input}

Provide your strategic analysis using the selected framework and always end with a Prescriptive Action.
"""}
                    ],
                    temperature=0.3
                )
                answer = response.choices[0].message.content
                st.write(answer)
            except Exception as e:
                st.error(f"LLM generation failed: {e}")
                answer = "Unable to generate strategic advice at this moment."
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
            except Exception as log_err:
                logger.warning(f"Logging failed: {log_err}")

# ============================================================
# STRATEGY PROPOSAL ROOM (unchanged)
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

st.session_state.messages = st.session_state.messages[-10:]
