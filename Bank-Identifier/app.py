import streamlit as st
import PyPDF2
import os
import webbrowser
import threading
import uuid
import subprocess
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Create temp directory if not exists
TEMP_DIR = "./temp"
os.makedirs(TEMP_DIR, exist_ok=True)


# Start simple HTTP server in a separate thread
def start_http_server():
    port = 8502
    handler = SimpleHTTPRequestHandler
    httpd = HTTPServer(('', port), handler)
    print(f"Serving HTTP on port {port}...")
    httpd.serve_forever()


# Start HTTP server in background thread
if not hasattr(st, 'http_server_started'):
    server_thread = threading.Thread(target=start_http_server, daemon=True)
    server_thread.start()
    st.http_server_started = True


def extract_text_from_pdf(pdf_file):
    text = ''
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text


def identify_bank_from_text(text):
    text_lower = text.lower()
    bank_keywords = {
        'Absa': ['absa'],
        'FNB': ['fnb', 'first national bank'],
        'Nedbank': ['nedbank'],
        'Standard Bank': ['standard bank'],
        'Capitec Bank': ['capitec']
    }

    for bank, keywords in bank_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return bank
    return None


# Bank to port mapping and app paths
BANK_APP_CONFIG = {
    'Absa': {
        'port': 8503,
        'path': '../ABSA Bank/app.py'
    },
    'FNB': {
        'port': 8504,
        'path': '../First National Bank/app.py'
    },
    'Nedbank': {
        'port': 8505,
        'path': '../NedBank/app.py'
    },
    'Standard Bank': {
        'port': 8506,
        'path': '../Standard Bank/app.py'
    },
    'Capitec Bank': {
        'port': 8507,
        'path': '../Capitec Bank/app.py'  # Add if exists
    }
}

# Track running bank apps
if 'running_banks' not in st.session_state:
    st.session_state.running_banks = {}


def launch_bank_app(bank_name):
    """Launch bank-specific Streamlit app in background"""
    config = BANK_APP_CONFIG[bank_name]
    cmd = [
        "streamlit",
        "run",
        config['path'],
        "--server.port",
        str(config['port']),
        "--server.headless",
        "true"
    ]

    # Set working directory to bank's folder
    cwd = os.path.dirname(config['path'])

    # Launch the app
    process = subprocess.Popen(cmd, cwd=cwd)
    st.session_state.running_banks[bank_name] = process.pid
    return f"http://localhost:{config['port']}/"


st.title("Bank Statement Analyzer")
uploaded_file = st.file_uploader("Upload Bank Statement PDF", type="pdf")

if uploaded_file is not None:
    # Extract text and identify bank
    pdf_text = extract_text_from_pdf(uploaded_file)
    bank_name = identify_bank_from_text(pdf_text)

    if bank_name is None:
        st.error("Bank not recognized")
    elif bank_name not in BANK_APP_CONFIG:
        st.error(f"No analyzer available for {bank_name}")
    else:
        st.success(f"Detected Bank: {bank_name}")

        # Save text to temp file
        session_id = str(uuid.uuid4())
        temp_file = os.path.join(TEMP_DIR, f"{session_id}.txt")
        with open(temp_file, "w") as f:
            f.write(pdf_text)

        # Launch bank app if not running
        if bank_name not in st.session_state.running_banks:
            with st.spinner(f"🚀 Launching {bank_name} analyzer..."):
                bank_url = launch_bank_app(bank_name)
                time.sleep(2)  # Allow app startup time
        else:
            config = BANK_APP_CONFIG[bank_name]
            bank_url = f"http://localhost:{config['port']}/"

        # Open bank app in new tab
        bank_url += f"?session_id={session_id}"
        js = f"window.open('{bank_url}')"

        st.markdown(f"""
        <div style="text-align:center; margin-top:30px">
            <button onclick="{js}" style="
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                transition: all 0.3s ease;
            ">
                ⚡️ Open {bank_name} Analyzer
            </button>
        </div>
        """, unsafe_allow_html=True)

        st.info(f"### 🔍 Analyzing {bank_name} Statement\n"
                "The analyzer will show:\n"
                "- Income vs Expense trends\n"
                "- Spending categories\n"
                "- Financial health score\n"
                "- Anomaly detection")

        # Creative visualization
        st.image("https://cdn.pixabay.com/photo/2018/05/18/15/30/web-design-3411373_960_720.jpg",
                 caption="Advanced Bank Analytics Dashboard")