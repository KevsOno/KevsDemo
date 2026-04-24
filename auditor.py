import os
import smtplib
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from supabase import create_client  # <--- CRITICAL LINE
from groq import Groq
import google.generativeai as genai

# --- 1. CONFIGURATION & CLIENT SETUP ---
CREDENTIALS = {
    "SB_URL": os.environ.get('SUPABASE_URL', '').strip().rstrip('/'),
    "SB_KEY": os.environ.get('SUPABASE_KEY'),
    "G_API": os.environ.get('GEMINI_API_KEY'),
    "GROQ_API": os.environ.get('GROQ_API_KEY'),
    "EMAIL_USER": os.environ.get('GMAIL_USER'),
    "EMAIL_PASS": os.environ.get('GMAIL_PASS')
}

# Now this line will work because 'create_client' is defined
supabase = create_client(CREDENTIALS["SB_URL"], CREDENTIALS["SB_KEY"])
