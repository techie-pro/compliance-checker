import streamlit as st
import PyPDF2
import requests
import json
from io import BytesIO

# OpenRouter API configuration
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = st.secrets["OPENROUTER_API_KEY"]
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
   
}
MODEL = "x-ai/grok-4-fast:free"

# FTC Guidelines Prompt Template
FTC_GUIDELINES = """
Analyze the following text for potential violations of US FTC Section 5 (Unfair or Deceptive Acts or Practices).

**Unfair Practices (3-part test):**
1. Does it cause substantial injury to consumers (e.g., financial loss, privacy harm)?
2. Is the injury reasonably avoidable by consumers themselves?
3. Is the injury outweighed by countervailing benefits (e.g., to competition or consumers)?

**Deceptive Practices:**
1. Representation, omission, or practice likely to mislead a reasonable consumer.
2. Material (likely affects purchase/conduct decisions).

Examples of violations: false/misleading claims (e.g., "guaranteed results" without basis), hidden fees, bait-and-switch, failure to disclose risks/hazards, unsubstantiated superiority claims.

For the text: Flag any potential issues with:
- Exact quote from text.
- Issue type (e.g., 'deceptive claim', 'unfair omission').
- Brief explanation.
- Suggested rephrasing to fix (make compliant).

If no issues, respond: "No issues found."
Respond in JSON: {"flags": [{"quote": "...", "issue_type": "...", "explanation": "...", "suggestion": "..."}]}
"""

def extract_pdf_text(uploaded_file):
    """Extract text from PDF page-by-page."""
    pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
    pages_text = []
    for page_num, page in enumerate(pdf_reader.pages, start=1):
        text = page.extract_text()
        pages_text.append({"page": page_num, "text": text.strip()})
    return pages_text

def call_openrouter(prompt, max_tokens=500):
    """Call OpenRouter API."""
    response = requests.post(
        API_URL,
        headers=HEADERS,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        st.error(f"API Error: {response.status_code} - {response.text}")
        return None

def analyze_pages(pages_text):
    """Analyze each page for flags."""
    all_flags = []
    clean_pages = 0
    for page_data in pages_text:
        prompt = f"{FTC_GUIDELINES}\n\nText (Page {page_data['page']}): {page_data['text'][:2000]}"  # Truncate if too long
        result = call_openrouter(prompt)
        if result:
            try:
                flags_json = json.loads(result)
                page_flags = flags_json.get("flags", [])
                all_flags.extend([{**f, "page": page_data["page"]} for f in page_flags])  # Fixed line
                if not page_flags:
                    clean_pages += 1
            except json.JSONDecodeError:
                st.warning(f"Invalid JSON response for page {page_data['page']}")
    return all_flags, clean_pages, len(pages_text)

def generate_summary(full_text):
    """Generate document summary."""
    prompt = f"Summarize the key points of this document concisely (200 words max): {full_text[:4000]}"
    return call_openrouter(prompt)

def generate_fixed_text(full_text, flags):
    """Generate fixed version aiming for 100% compliance."""
    issues_summary = "\n".join([f"Page {f['page']}: {f['explanation']}" for f in flags]) if flags else "No issues."
    prompt = f"""Rewrite the following document to eliminate all unfair/deceptive elements per FTC guidelines.
Issues to address: {issues_summary}

Original text: {full_text}

Output only the rewritten full text, maintaining structure and length as much as possible."""
    return call_openrouter(prompt, max_tokens=2000)

# Streamlit UI
st.title("PDF FTC Compliance Checker")
st.write("Upload a PDF to analyze for unfair/deceptive practices under US FTC Section 5.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if st.button("Analyze Document"):
        with st.spinner("Extracting and analyzing..."):
            pages_text = extract_pdf_text(uploaded_file)
            full_text = " ".join([p["text"] for p in pages_text])
            
            # Analyze
            flags, clean_pages, total_pages = analyze_pages(pages_text)
            
            # Score
            score = max(0, 100 - ((total_pages - clean_pages) * 10))  # 10% deduction per flagged page
            st.metric("Compliance Score", f"{score}%", delta=None)
            
            if score < 100:
                st.warning("Document has potential issues. See flags below.")
            
            # Summary
            st.subheader("Document Summary")
            summary = generate_summary(full_text)
            if summary:
                st.write(summary)
            
            # Flags
            if flags:
                st.subheader("Flagged Issues")
                for flag in flags:
                    with st.expander(f"Page {flag['page']} - {flag['issue_type']}"):
                        st.write("**Quote:** " + flag["quote"])
                        st.write("**Explanation:** " + flag["explanation"])
                        st.write("**Suggested Fix:** " + flag["suggestion"])
            else:
                st.success("No flags found!")
            
            # Fixed Version
            st.subheader("Fixed Version (100% Compliant Rewrite)")
            fixed_text = generate_fixed_text(full_text, flags)
            if fixed_text:
                st.text_area("Rewritten Document", fixed_text, height=300)
            else:
                st.error("Failed to generate fixed version.")