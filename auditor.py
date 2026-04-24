import os  # Add this to your imports at the top

# --- 1. CONFIGURATION & CLIENT SETUP ---
# os.environ.get() looks for the Secrets you just added to your GitHub Settings
CREDENTIALS = {
    "SB_URL": os.environ.get('SUPABASE_URL', '').strip().rstrip('/'),
    "SB_KEY": os.environ.get('SUPABASE_KEY'),
    "G_API": os.environ.get('GEMINI_API_KEY'),
    "GROQ_API": os.environ.get('GROQ_API_KEY'),
    "EMAIL_USER": os.environ.get('GMAIL_USER'),
    "EMAIL_PASS": os.environ.get('GMAIL_PASS')
}

# Add a quick check to prevent crashes if a key is missing
for key, value in CREDENTIALS.items():
    if not value:
        print(f"⚠️ Warning: {key} is not set in environment variables.")

supabase = create_client(CREDENTIALS["SB_URL"], CREDENTIALS["SB_KEY"])
groq_client = Groq(api_key=CREDENTIALS["GROQ_API"])
genai.configure(api_key=CREDENTIALS["G_API"])
