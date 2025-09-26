import streamlit as st
import fitz  # PyMuPDF
import paddleocr
import requests
import json
import time
from io import BytesIO
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from collections import defaultdict

# OpenRouter API configuration
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = <apikey_placeholder>
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}
MODEL = "x-ai/grok-4-fast:free"

# PaddleOCR setup
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="en")

# Default custom rules if rules.json is missing
DEFAULT_CUSTOM_RULES = [
    {
        "name": "Deceptive Claims",
        "description": "Prohibit misleading claims like 'guaranteed approval' or 'everyone approved'.",
        "severity": "Major",
        "pattern": "guaranteed approval|everyone approved|always approved",
        "remediation": "Qualify claims with conditions (e.g., 'Subject to credit approval').",
        "risk": "Consumer deception leading to complaints or legal action.",
        "penalties": "FTC fines up to $51,744 per violation."
    },
    {
        "name": "Abusive Contract Terms",
        "description": "Prohibit non-cancellable contracts or terms limiting consumer rights.",
        "severity": "Critical",
        "pattern": "can’t cancel|non-cancellable|no cancellations",
        "remediation": "Include clear cancellation policies compliant with UDAAP.",
        "risk": "Consumer harm and regulatory scrutiny.",
        "penalties": "CFPB fines up to $1M per violation."
    }
]

# Load custom business rules
try:
    with open("rules.json", "r") as f:
        CUSTOM_RULES = json.load(f)
    st.info("Loaded rules.json successfully.")
except FileNotFoundError:
    CUSTOM_RULES = DEFAULT_CUSTOM_RULES
    st.warning("rules.json not found. Using default rules to detect deceptive/abusive terms. Create rules.json in the same directory as newModel.py to customize.")

# Comprehensive Regulatory Prompt
REGULATORY_GUIDELINES = """
Analyze the text for violations of U.S. federal and state regulations, focusing on deceptive or abusive language:

1. **Federal Banking**:
   - OCC: Responsible innovation (12 CFR Part 7).
   - FDIC: Consumer protection compliance (12 CFR Part 328).
   - Federal Reserve: Community Reinvestment Act (12 CFR Part 228).
2. **Consumer Protection**:
   - FTC Section 5: Unfair (substantial, unavoidable injury not outweighed by benefits) or Deceptive (misleading, material) acts (15 U.S.C. § 45). Flag terms like "guaranteed approval" or "everyone approved" unless clearly qualified.
   - UDAAP: Unfair, deceptive, or abusive acts (Dodd-Frank Act, 12 U.S.C. § 5531). Flag terms like "can’t cancel" or "non-cancellable" as abusive.
   - FCRA: Inaccurate credit reporting or improper data use (15 U.S.C. § 1681).
3. **Fair Lending**:
   - ECOA: Discrimination in credit (15 U.S.C. § 1691).
   - FHA: Housing discrimination (42 U.S.C. § 3601).
4. **Privacy**:
   - GLBA: Privacy Rule (consumer data sharing, 15 U.S.C. § 6801) and Safeguards Rule (data security, 16 CFR Part 314).
   - CCPA/CPRA: Consumer data rights (Cal. Civ. Code § 1798.100).
   - VCDPA: Virginia data protection (Va. Code § 59.1-575).
   - CPA: Colorado Privacy Act (Colo. Rev. Stat. § 6-1-1301).
5. **Communications**:
   - TCPA: Unsolicited calls/texts (47 U.S.C. § 227).
   - CAN-SPAM: Email marketing violations (15 U.S.C. § 7701).
6. **Truth in Advertising**: FTC Endorsement Guidelines (16 CFR Part 255).
7. **Banking Disclosures**:
   - Regulation DD: Truth in Savings (12 CFR Part 1030).
   - Regulation Z: Truth in Lending (12 CFR Part 1026).

Custom Rules:
{0}

For each chunk, return JSON:
{{
  "flags": [
    {{
      "regulation": "regulation name and section (e.g., FTC Section 5, 15 U.S.C. § 45)",
      "severity": "Critical/Major/Minor",
      "quote": "exact text",
      "explanation": "why it violates",
      "remediation": "specific fix",
      "risk": "potential impact",
      "penalties": "possible fines/consequences"
    }}
  ]
}}
If no issues: {{"flags": []}}
Severity:
- Critical: High consumer harm, unavoidable, no benefits (e.g., fraud, data breach, non-cancellable terms).
- Major: Misleading material claims, moderate harm (e.g., hidden fees, deceptive guarantees).
- Minor: Technical violations, low harm (e.g., missing disclosure).
"""

def extract_pdf_text(uploaded_file):
    """Extract text using PyMuPDF; fall back to PaddleOCR for scanned pages."""
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    pages_text = []
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text("text").strip()
        if not text:  # Scanned page
            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
            img = BytesIO(pix.tobytes())
            ocr_result = ocr.ocr(img.read(), cls=True)
            text = "\n".join(line[1][0] for line in ocr_result[0]) if ocr_result else ""
        pages_text.append({"page": page_num + 1, "text": text})
    pdf.close()
    return pages_text

def call_openrouter(prompt, max_tokens=1000, max_retries=3):
    """Call OpenRouter API with retry logic."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL,
                headers=HEADERS,
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 401:
                st.error("API Error 401: Invalid OpenRouter API key. Verify at openrouter.ai/keys.")
                return None
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except requests.RequestException as e:
            if "NameResolutionError" in str(e) or "getaddrinfo failed" in str(e):
                st.warning(f"Network/DNS issue (attempt {attempt + 1}/{max_retries}). Retrying in {2 ** attempt}s...")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            st.error(f"API Request Failed: {e}")
            return None
    return None

def analyze_pages(pages_text):
    """Analyze pages with semantic chunking and deduplication, batching chunks per page."""
    chunker = SemanticChunker(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=0.9  # Smaller chunks for better violation detection
    )
    all_flags = []
    violation_counts = defaultdict(int)
    for page_data in pages_text:
        if not page_data["text"]:
            continue
        chunks = chunker.split_text(page_data["text"])
        # Batch chunks to reduce API calls (max 4000 chars to stay under Grok's limit)
        batch_prompt = ""
        for i, chunk in enumerate(chunks):
            chunk_text = chunk[:2000]
            batch_prompt += f"\n\nChunk {i+1} (Page {page_data['page']}): {chunk_text}"
        if batch_prompt:
            prompt = f"{REGULATORY_GUIDELINES.format(json.dumps(CUSTOM_RULES))}\n\nText:{batch_prompt}"
            result = call_openrouter(prompt, max_tokens=1500)  # Increased tokens for batch
            if result:
                try:
                    flags_json = json.loads(result)
                    for f in flags_json.get("flags", []):
                        violation_key = (f["regulation"], f["quote"], f["severity"])
                        if violation_counts[violation_key] == 0:
                            f["page"] = page_data["page"]  # Ensure page number is set
                            all_flags.append(f)
                        violation_counts[violation_key] += 1
                except json.JSONDecodeError:
                    st.warning(f"Invalid JSON response for page {page_data['page']}")
    
    # Calculate score
    penalties = {"Critical": 40, "Major": 20, "Minor": 5}
    score = 100
    for violation_key, count in violation_counts.items():
        severity = violation_key[2]
        score -= penalties[severity]
    score = max(0, score)
    
    return all_flags, score, len(pages_text)

def generate_summary(full_text, flags):
    """Generate executive summary with key findings."""
    violation_summary = "\n".join([f"Page {f['page']}: {f['regulation']} ({f['severity']}) - {f['explanation']}" for f in flags]) if flags else "No violations found."
    prompt = f"""Provide an executive summary (150 words or less) of the document's key points and compliance issues:
Document: {full_text[:4000]}
Violations: {violation_summary}"""
    return call_openrouter(prompt, max_tokens=200)

def generate_json_report(flags, score, total_pages, summary):
    """Generate JSON report."""
    return {
        "score": score,
        "total_pages": total_pages,
        "violations": flags,
        "executive_summary": summary
    }

# Streamlit UI
st.title("PDF Regulatory Compliance Checker")
st.write("Upload a PDF to analyze for federal/state regulatory violations.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
rules_uploaded = st.file_uploader("Upload custom rules (JSON)", type="json")

if rules_uploaded:
    try:
        CUSTOM_RULES = json.load(rules_uploaded)
        st.info("Custom rules uploaded successfully.")
    except json.JSONDecodeError:
        st.error("Invalid JSON in rules file. Using default rules.")
        CUSTOM_RULES = DEFAULT_CUSTOM_RULES

if uploaded_file:
    if st.button("Analyze Document"):
        with st.spinner("Extracting and analyzing..."):
            # Extract text
            pages_text = extract_pdf_text(uploaded_file)
            full_text = " ".join([p["text"] for p in pages_text if p["text"]])
            
            if not full_text.strip():
                st.error("No text extracted from PDF.")
                st.stop()
            
            # Analyze
            flags, score, total_pages = analyze_pages(pages_text)
            
            # Summary
            summary = generate_summary(full_text, flags)
            
            # JSON Report
            st.subheader("Compliance Report")
            report = generate_json_report(flags, score, total_pages, summary)
            st.json(report)
            
            # UI Details
            st.metric("Compliance Score", f"{score}%", delta=f"{score-100}%")
            if score < 100:
                st.warning(f"Violations found on {len(set(f['page'] for f in flags))} page(s).")
            
            st.subheader("Executive Summary")
            if summary:
                st.write(summary)
            else:
                st.error("Failed to generate summary.")
            
            if flags:
                st.subheader("Violations")
                for flag in flags:
                    with st.expander(f"Page {flag['page']} - {flag['regulation']} ({flag['severity']})"):
                        st.write(f"**Quote:** {flag['quote']}")
                        st.write(f"**Explanation:** {flag['explanation']}")
                        st.write(f"**Remediation:** {flag['remediation']}")
                        st.write(f"**Risk:** {flag['risk']}")
                        st.write(f"**Penalties:** {flag['penalties']}")
            else:
                st.success("No violations found!")