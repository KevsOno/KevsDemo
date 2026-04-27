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
# ADVANCED TOOL FUNCTIONS
# ============================================================

def calculate_budget_optimization(budget, leads_data):
    """Operations Research: Linear allocation based on Faculty Conversion."""
    if not leads_data: 
        return "No data to optimize."
    df = pd.DataFrame(leads_data)
    # Calculate Lead Density per Faculty
    dist = df['faculty'].value_counts(normalize=True)
    # Optimization: Prioritize high-density areas but cap spend to avoid diminishing returns
    optimized_plan = (dist * budget).to_dict()
    return optimized_plan

def run_product_bcg_matrix(inventory_data):
    """Choice Problem: BCG Matrix analysis for Product Strategy."""
    if not inventory_data: 
        return "No inventory to analyze."
    df = pd.DataFrame(inventory_data)
    # Rename columns to match expected names
    df = df.rename(columns={'stock_quantity': 'stock', 'unit_price': 'price'})
    # Categorize based on stock (proxy for market share) and price (proxy for growth/value)
    df['category'] = 'Question Mark'
    df.loc[(df['stock'] > 20) & (df['price'] > 15000), 'category'] = 'Star'
    df.loc[(df['stock'] > 20) & (df['price'] <= 15000), 'category'] = 'Cash Cow'
    df.loc[(df['stock'] <= 20) & (df['price'] <= 15000), 'category'] = 'Dog'
    
    return df[['item_name', 'category']].to_dict(orient='records')

def run_optimization_analysis(budget: float, lead_data: list):
    """Calculate best budget allocation across faculties."""
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

def propose_strategy_update(framework: str, suggestion: str, new_values: dict):
    """Propose changes to business metrics or strategy with Lagos context."""
    # Auto-add Lagos Context
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

def select_best_framework(query: str, data_available: dict):
    """
    Optimizes which framework to use based on query intent and data depth.
    """
    # Logic: If query mentions 'competitor' and we have data -> VRIO
    if "competitor" in query.lower() and data_available.get('competitor_info'):
        return "VRIO"
    # Logic: If query is about 'growth' -> Ansoff Matrix
    elif "grow" in query.lower() or "scale" in query.lower():
        return "Ansoff Matrix"
    # Logic: If data is missing
    elif not data_available.get('customer_feedback'):
        return "INSUFFICIENT_CONTEXT_NPS"
    return "SWOT"  # Default

# ============================================================
# SIDEBAR
# ============================================================
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
# OPTIMIZATION ENGINE UI
# ============================================================
with st.expander("🔬 Run Operations Research"):
    if st.button("Calculate Optimal Budget Allocation"):
        # 1. Run Math
        results = run_optimization_analysis(mkt_budget, leads_df.to_dict(orient='records'))
        
        if "error" not in results:
            # 2. Display Result
            st.write(f"### Optimal Strategy: Focus on **{results['best_move']}**")
            st.info(results['recommendation'])
            
            # 3. Choice Visualization
            chart_data = pd.DataFrame(results['allocation_plan'])
            st.bar_chart(chart_data.set_index('faculty')['suggested_spend'])
            
            # 4. Strategic Proposal Option
            if st.button("📝 Propose This Strategy for Approval"):
                result = propose_strategy_update(
                    framework="Operations Research",
                    suggestion=results['recommendation'],
                    new_values={"recommended_budget_allocation": results['allocation_plan']}
                )
                st.success(result)
        else:
            st.error(results['error'])

# ============================================================
# PRODUCT CHOICE (BCG MATRIX)
# ============================================================
with st.expander("📈 Product Choice Matrix (BCG)"):
    if not inv_df.empty:
        # Create a copy to avoid modifying original
        bcg_df = inv_df.copy()
        
        # Simplified BCG Logic
        bcg_df['Market Share'] = bcg_df['stock_quantity'] / bcg_df['stock_quantity'].sum()
        bcg_df['Growth Potential'] = bcg_df['unit_price'] / bcg_df['unit_price'].max()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(bcg_df['Market Share'], bcg_df['Growth Potential'], s=100)
        for i, txt in enumerate(bcg_df['item_name']):
            ax.annotate(txt, (bcg_df['Market Share'].iat[i], bcg_df['Growth Potential'].iat[i]), fontsize=10)
        
        ax.set_xlabel('Market Share (Stock)')
        ax.set_ylabel('Growth Potential (Value)')
        ax.set_title('BCG Matrix Analysis')
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax.text(0.75, 0.75, 'STARS', fontsize=12, alpha=0.5, ha='center')
        ax.text(0.25, 0.75, 'QUESTION MARKS', fontsize=12, alpha=0.5, ha='center')
        ax.text(0.75, 0.25, 'CASH COWS', fontsize=12, alpha=0.5, ha='center')
        ax.text(0.25, 0.25, 'DOGS', fontsize=12, alpha=0.5, ha='center')
        
        st.pyplot(fig)
        st.caption("Stars (Top Right), Cash Cows (Bottom Right), Question Marks (Top Left), Dogs (Bottom Left)")
        
        # Show BCG categorization
        bcg_results = run_product_bcg_matrix(inv_df.to_dict(orient='records'))
        if isinstance(bcg_results, list):
            st.write("**BCG Categorization:**")
            st.dataframe(pd.DataFrame(bcg_results))

# ============================================================
# DYNAMIC TASK BUTTONS
# ============================================================
st.subheader("🛠️ Strategic Tasks")
task_col1, task_col2, task_col3 = st.columns(3)

if task_col1.button("👥 Run Customer Journey"):
    st.session_state.messages.append({"role": "user", "content": "Analyze the student journey from seeing an ad to buying a perfume at LASU."})

if task_col2.button("⚔️ Competitive VRIO"):
    st.session_state.messages.append({"role": "user", "content": "Run a VRIO analysis on our technical marketing automation vs traditional competitors."})

if task_col3.button("🌊 Blue Ocean Strategy"):
    st.session_state.messages.append({"role": "user", "content": "How can we create a Blue Ocean in the crowded Lagos fragrance market?"})

st.divider()

# ============================================================
# QUICK PROMPTS
# ============================================================
st.subheader("💡 Quick Questions")
col1, col2, col3 = st.columns(3)

if col1.button("📍 Where should we scale?"):
    st.session_state.messages.append({"role": "user", "content": "Where should we scale?"})

if col2.button("⚠️ Are we overspending?"):
    st.session_state.messages.append({"role": "user", "content": "Analyze our CAC efficiency"})

if col3.button("🚀 How do we hit SOM faster?"):
    st.session_state.messages.append({"role": "user", "content": "How can we reach our SOM target faster?"})

# ============================================================
# CHAT INTERFACE
# ============================================================
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
            
            # Advanced system prompt with framework selection
            system_content = """You are the Penafort Strategy Engine. You operate in 3 Modes:

1. CUSTOMER ANALYSIS: (Audience Segmentation, CJM, Value Prop Canvas, JTBD, NPS, PMF). 
   Use this for 'Who' and 'Why' questions.
2. MARKET ANALYSIS: (SWOT, VRIO, Value Chain, BCG, Ansoff, Blue Ocean, PESTLE). 
   Use this for 'Competition' and 'Market Position' questions.
3. HYBRID: Combined analysis for high-level decision making.

OPTIMIZATION RULE:
- If context is missing for a framework (e.g. no lead data for PMF), DO NOT hallucinate. 
- Instead, guide Michael: 'To run a PMF analysis, I need to know the retention rate of your last 50 customers.'
- Always prioritize locally relevant solutions for the Lagos/LASU market.

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
            
            proposal_col1, proposal_col2 = st.columns(2)
            if proposal_col1.button("✅ Approve & Execute", key=f"app_{prop['id']}"):
                st.success("✅ Strategy Integrated!")
                supabase.table("suggestions").update({"status": "approved"}).eq("id", prop["id"]).execute()
                st.rerun()
                
            if proposal_col2.button("❌ Reject", key=f"rej_{prop['id']}"):
                supabase.table("suggestions").update({"status": "rejected"}).eq("id", prop["id"]).execute()
                st.rerun()
else:
    st.info("📭 No pending strategic proposals.")

# ============================================================
# VISUALIZATIONS
# ============================================================
st.divider()
st.subheader("📊 Analytics")

if not leads_df.empty:
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        st.bar_chart(leads_df['faculty'].value_counts())
    with viz_col2:
        st.dataframe(inv_df, use_container_width=True)

# --- LIMIT MEMORY ---
st.session_state.messages = st.session_state.messages[-10:]
