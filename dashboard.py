import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from supabase import create_client
import google.generativeai as genai
from groq import Groq
from datetime import datetime
import json

# ... (Previous Config & Clients remain the same) ...

# ============================================================
# 1. INTELLIGENT INTENT CLASSIFIER (Replaces Keyword Matching)
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
    except:
        return "HYBRID_OPTIMIZATION" # Fallback

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
            "missing_msg": "⚠️ Customer analysis requires lead data. Ask Michael to sync leads."
        },
        "MARKET_ANALYSIS": {
            "pillar": 2,
            "frameworks": ["VRIO", "Blue Ocean", "PESTLE"],
            "data_check": True, # Usually external data
            "missing_msg": "🌍 Market analysis active - using competitive intelligence."
        },
        "HYBRID_OPTIMIZATION": {
            "pillar": 3,
            "frameworks": ["Ansoff Matrix", "Resource Allocation"],
            "data_check": has_leads and has_inventory,
            "missing_msg": "🔗 Connecting customer needs to market opportunities."
        }
    }
    
    selected = config.get(intent, config["HYBRID_OPTIMIZATION"])
    status = "READY" if selected["data_check"] else "DATA_MISSING"
    
    return {
        "mode": intent,
        "status": status,
        "message": selected["missing_msg"] if not selected["data_check"] else f"✅ {intent.replace('_', ' ')} Ready",
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
    
    context_block = "\n".join([f"ID:{i} | Content:{m['content']}" for i, m in enumerate(memories)])
    
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
            temperature=0
        )
        relevant_ids = response.choices[0].message.content.strip().split(',')
        if "NONE" in relevant_ids:
            return ""
        
        filtered_content = [memories[int(idx)]['content'] for idx in relevant_ids if idx.strip().isdigit()]
        return "\n---\n".join(filtered_content)
    except:
        return memories[0]['content'] # Fallback to top vector match

# ============================================================
# 3. SPECIALIZED SYSTEM PROMPTS (Replaces Monolithic Prompt)
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
    
    return f"""You are {p['role']} at Penafort Strategic.
    
OPERATIONAL PHILOSOPHY:
- {p['strategy']}
- Tone: {p['tone']}
- Location Context: {user_location} (specifically the {user_niche}).

CONSTRAINTS:
1. Only use the frameworks: {', '.join(framework_result['frameworks'])}.
2. If data is missing (Status: {framework_result['status']}), stop and ask Michael for specific inputs.
3. Always provide a 'Prescriptive Action' at the end of your response.
"""

# ... (The rest of the Chat logic uses these new functions) ...
