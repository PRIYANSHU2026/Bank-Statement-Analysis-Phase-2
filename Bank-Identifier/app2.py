import streamlit as st
import PyPDF2
import os
import webbrowser
import threading
import uuid
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


# Bank to port mapping
BANK_PORT_MAPPING = {
    'Absa': 8503,
    'FNB': 8504,
    'Nedbank': 8505,
    'Standard Bank': 8506,
    'Capitec Bank': 8507
}

st.title("Bank Statement Analyzer")
uploaded_file = st.file_uploader("Upload Bank Statement PDF", type="pdf")

if uploaded_file is not None:
    # Extract text and identify bank
    pdf_text = extract_text_from_pdf(uploaded_file)
    bank_name = identify_bank_from_text(pdf_text)

    if bank_name is None:
        st.error("Bank not recognized")
    elif bank_name not in BANK_PORT_MAPPING:
        st.error(f"No analyzer available for {bank_name}")
    else:
        st.success(f"Detected Bank: {bank_name}")

        # Save text to temp file
        session_id = str(uuid.uuid4())
        temp_file = os.path.join(TEMP_DIR, f"{session_id}.txt")
        with open(temp_file, "w") as f:
            f.write(pdf_text)

        # Get port for bank-specific app
        port = BANK_PORT_MAPPING[bank_name]
        bank_url = f"http://localhost:{port}/?session_id={session_id}"

        # Open bank app in new tab
        st.markdown(f"""
        <div style="text-align:center; margin-top:30px">
            <a href="{bank_url}" target="_blank">
                <button style="
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
                ">
                    Open {bank_name} Analyzer in New Tab
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)

        st.info(f"Click the button above to open the {bank_name} analyzer in a new tab")