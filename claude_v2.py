import streamlit as st
import fitz  # PyMuPDF
import paddleocr
import requests
import json
import time
from io import BytesIO
import base64
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from collections import defaultdict, Counter
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import hashlib

# OpenRouter API configuration
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = st.secrets["OPENROUTER_API_KEY"]  # Replace with your actual API key
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}
MODEL = "x-ai/grok-4-fast:free"

# Enhanced page config
st.set_page_config(
    page_title="Compliance Document Analyzer Pro",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2d5aa0 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .violation-card {
        border-left: 4px solid #ff4444;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #fafafa;
        border-radius: 0 8px 8px 0;
    }
    
    .violation-card.critical {
        border-left-color: #dc3545;
        background: #fff5f5;
    }
    
    .violation-card.major {
        border-left-color: #ffc107;
        background: #fffbf0;
    }
    
    .violation-card.minor {
        border-left-color: #17a2b8;
        background: #f0f9ff;
    }
    
    .pdf-container {
        height: 600px;
        border: 1px solid #ddd;
        border-radius: 8px;
        overflow: hidden;
    }
    
    .json-viewer {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        max-height: 500px;
        overflow-y: auto;
    }
    
    .section-header {
        background: #f8f9fa;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #007bff;
    }
    
    .progress-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# PaddleOCR setup with compatible settings
@st.cache_resource
def load_ocr():
    try:
        return paddleocr.PaddleOCR(use_angle_cls=True, lang="en")
    except Exception as e:
        st.error(f"Failed to initialize PaddleOCR: {e}")
        try:
            return paddleocr.PaddleOCR(lang="en")
        except Exception as e2:
            st.error(f"Failed to initialize basic OCR: {e2}")
            return None

ocr = load_ocr()

@dataclass
class Violation:
    regulation: str
    severity: str
    quote: str
    explanation: str
    remediation: str
    risk: str
    penalties: str
    page: int
    chunk_id: str
    confidence: float = 1.0

    def to_dict(self):
        return asdict(self)

class ComplianceAnalyzer:
    def __init__(self):
        self.violation_cache = {}
        self.load_custom_rules()
        self.setup_text_splitter()
    
    def load_custom_rules(self):
        """Load custom rules with enhanced default ruleset"""
        self.DEFAULT_CUSTOM_RULES = [
            {
                "name": "Deceptive Guarantee Claims",
                "description": "Prohibit misleading guarantee claims without proper qualification",
                "severity": "Major",
                "patterns": [
                    "guaranteed approval", "100% approval", "everyone approved", "always approved",
                    "guaranteed returns", "risk-free", "can't lose", "certain profit"
                ],
                "remediation": "Qualify all claims with appropriate conditions and disclaimers",
                "risk": "FTC enforcement action for deceptive advertising",
                "penalties": "Up to $51,744 per violation under FTC Act"
            },
            {
                "name": "Abusive Contract Terms",
                "description": "Identify potentially abusive or unfair contract terms",
                "severity": "Critical", 
                "patterns": [
                    "cannot cancel", "non-cancellable", "no refunds", "binding forever",
                    "waive all rights", "no legal recourse", "mandatory arbitration"
                ],
                "remediation": "Include fair cancellation policies and consumer rights protection",
                "risk": "CFPB enforcement for UDAAP violations",
                "penalties": "Up to $1M per day for ongoing violations"
            },
            {
                "name": "Hidden Fee Violations",
                "description": "Detect undisclosed or hidden fees",
                "severity": "Major",
                "patterns": [
                    "no fees*", "free*", "additional charges may apply", "other fees",
                    "administrative fee", "processing fee", "handling charge"
                ],
                "remediation": "Provide clear, conspicuous disclosure of all fees upfront",
                "risk": "TILA violations and consumer complaints",
                "penalties": "Statutory damages and attorney fees"
            },
            {
                "name": "Discriminatory Language",
                "description": "Flag potentially discriminatory language",
                "severity": "Critical",
                "patterns": [
                    "ideal family", "perfect credit", "excellent neighborhood", "quality tenants",
                    "mature individuals", "stable family", "traditional values"
                ],
                "remediation": "Use neutral, non-discriminatory language in all communications",
                "risk": "Fair Housing Act and ECOA violations",
                "penalties": "HUD penalties up to $21,039 first offense"
            },
            {
                "name": "Privacy Violations",
                "description": "Identify privacy and data protection issues",
                "severity": "Major",
                "patterns": [
                    "share your information", "sell your data", "third party access",
                    "no privacy protection", "public information"
                ],
                "remediation": "Implement comprehensive privacy notices and data protection measures",
                "risk": "State privacy law violations and regulatory fines",
                "penalties": "Up to $7,500 per violation under CCPA"
            }
        ]
        
        try:
            with open("rules.json", "r") as f:
                self.CUSTOM_RULES = json.load(f)
            st.success("✅ Custom rules loaded successfully")
        except FileNotFoundError:
            self.CUSTOM_RULES = self.DEFAULT_CUSTOM_RULES
            # st.info("📋 Using enhanced default compliance rules")

    def setup_text_splitter(self):
        """Setup optimized text splitter for regulatory analysis"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
            keep_separator=True
        )

    def get_comprehensive_prompt(self) -> str:
        """Enhanced regulatory prompt with comprehensive coverage"""
        custom_rules_text = json.dumps(self.CUSTOM_RULES, indent=2)
        
        return f"""
You are an expert Federal Compliance Validator specializing in financial services regulations. Analyze the provided text for regulatory violations with precision and context awareness.

## REGULATORY FRAMEWORK

### FEDERAL BANKING REGULATIONS:
- **OCC Guidelines**: Responsible innovation, safety and soundness (12 CFR Part 7)
- **FDIC Rules**: Consumer protection, deposit insurance (12 CFR Part 328)  
- **Federal Reserve**: Community Reinvestment Act, bank supervision (12 CFR Part 228)
- **Regulation DD**: Truth in Savings - deposit account disclosures (12 CFR Part 1030)
- **Regulation Z**: Truth in Lending - credit cost disclosures (12 CFR Part 1026)

### CONSUMER PROTECTION LAWS:
- **FTC Section 5**: Unfair or Deceptive Acts and Practices (15 U.S.C. § 45)
  - Deceptive: Misleading + Material + Reasonable consumer interpretation
  - Unfair: Substantial injury + Unavoidable + Not outweighed by benefits
- **UDAAP**: Unfair, Deceptive, or Abusive Acts (12 U.S.C. § 5531)
  - Abusive: Takes advantage of consumer lack of understanding or inability to protect interests
- **FCRA**: Fair Credit Reporting Act - credit reporting accuracy (15 U.S.C. § 1681)

### FAIR LENDING LAWS:
- **ECOA**: Equal Credit Opportunity Act - prohibited discrimination (15 U.S.C. § 1691)
- **FHA**: Fair Housing Act - housing discrimination (42 U.S.C. § 3601)

### PRIVACY REGULATIONS:
- **GLBA Privacy Rule**: Consumer financial information sharing (15 U.S.C. § 6801)
- **GLBA Safeguards Rule**: Data security requirements (16 CFR Part 314)
- **CCPA/CPRA**: California consumer data rights (Cal. Civ. Code § 1798.100)
- **VCDPA**: Virginia Consumer Data Protection Act (Va. Code § 59.1-575)
- **Colorado Privacy Act**: Consumer data protection (Colo. Rev. Stat. § 6-1-1301)

### COMMUNICATIONS LAWS:
- **TCPA**: Telephone Consumer Protection Act - robocalls/texts (47 U.S.C. § 227)
- **CAN-SPAM Act**: Commercial email requirements (15 U.S.C. § 7701)
- **FDCPA**: Fair Debt Collection Practices Act (15 U.S.C. § 1692)

### ADVERTISING LAWS:
- **FTC Endorsement Guidelines**: Truth in advertising (16 CFR Part 255)
- **FTC Truth in Advertising**: Substantiation requirements

## CUSTOM BUSINESS RULES:
{custom_rules_text}

## SEVERITY CLASSIFICATION:
- **Critical**: High consumer harm, regulatory enforcement likely, significant penalties
- **Major**: Moderate harm, clear violation, substantial penalties possible  
- **Minor**: Technical violation, low harm, minimal penalties

## ANALYSIS INSTRUCTIONS:
1. Read text carefully for regulatory violations
2. Consider context and reasonable consumer interpretation
3. Quote exact problematic text - no paraphrasing
4. Provide specific regulation and section references
5. Give actionable remediation advice

## REQUIRED JSON OUTPUT:
{{
  "flags": [
    {{
      "regulation": "Specific law and section (e.g., 'FTC Section 5, 15 U.S.C. § 45')",
      "severity": "Critical|Major|Minor",
      "quote": "exact problematic text from document",
      "explanation": "detailed explanation of why this violates the regulation",
      "remediation": "specific actionable steps to fix the violation", 
      "risk": "potential regulatory and business risks",
      "penalties": "specific penalties and fines that could apply",
      "confidence": 0.95
    }}
  ]
}}

If no violations found: {{"flags": []}}

Be thorough but avoid false positives. Focus on clear, substantive violations that pose real regulatory risk.
"""

    def extract_pdf_text(self, uploaded_file) -> List[Dict]:
        """Enhanced PDF text extraction with better error handling"""
        try:
            pdf_bytes = uploaded_file.read()
            pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages_text = []
            
            progress_container = st.container()
            with progress_container:
                st.markdown('<div class="progress-container">', unsafe_allow_html=True)
                progress_bar = st.progress(0)
                progress_text = st.empty()
                st.markdown('</div>', unsafe_allow_html=True)
            
            total_pages = len(pdf)
            
            for page_num in range(total_pages):
                page = pdf[page_num]
                progress_text.text(f"Processing page {page_num + 1} of {total_pages}...")
                
                # Try text extraction first
                text = page.get_text("text").strip()
                
                # Enhanced OCR fallback for scanned pages
                if len(text) < 50 and ocr is not None:
                    try:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                        img_data = pix.tobytes("png")
                        
                        ocr_result = ocr.ocr(img_data, cls=True)
                        if ocr_result and ocr_result[0]:
                            ocr_texts = []
                            for line in ocr_result[0]:
                                if len(line) > 1 and line[1][1] > 0.7:
                                    ocr_texts.append(line[1][0])
                            text = "\n".join(ocr_texts)
                    except Exception as e:
                        st.warning(f"OCR failed for page {page_num + 1}: {str(e)}")
                        text = ""
                elif len(text) < 50 and ocr is None:
                    st.warning(f"Page {page_num + 1} appears to be scanned but OCR is not available")
                
                pages_text.append({
                    "page": page_num + 1,
                    "text": text.strip()
                })
                
                progress_bar.progress((page_num + 1) / total_pages)
            
            pdf.close()
            progress_bar.empty()
            progress_text.empty()
            progress_container.empty()
            
            return pages_text
            
        except Exception as e:
            st.error(f"Error extracting PDF: {str(e)}")
            return []

    def create_violation_hash(self, regulation: str, quote: str, severity: str) -> str:
        """Create hash for violation deduplication"""
        content = f"{regulation}||{quote.lower().strip()}||{severity}"
        return hashlib.md5(content.encode()).hexdigest()

    def call_openrouter(self, prompt: str, max_tokens: int = 1500, max_retries: int = 3) -> Optional[str]:
        """Enhanced API call with better error handling"""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    API_URL,
                    headers=HEADERS,
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.1,
                    },
                    timeout=90
                )
                
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"].strip()
                elif response.status_code == 401:
                    st.error("🔑 API Error: Invalid OpenRouter API key")
                    return None
                elif response.status_code == 429:
                    wait_time = 2 ** attempt
                    st.warning(f"⏱️ Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    st.error(f"❌ API Error {response.status_code}: {response.text}")
                    return None
                    
            except requests.RequestException as e:
                wait_time = 2 ** attempt
                st.warning(f"🌐 Network error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                return None
                
        return None

    def analyze_chunk(self, chunk_text: str, page_number: int, chunk_id: str) -> List[Violation]:
        """Analyze individual chunk for violations"""
        prompt = f"{self.get_comprehensive_prompt()}\n\nTEXT TO ANALYZE:\n{chunk_text}"
        
        result = self.call_openrouter(prompt)
        violations = []
        
        if result:
            try:
                result_clean = re.sub(r'^```json\s*', '', result.strip())
                result_clean = re.sub(r'\s*```$', '', result_clean)
                
                flags_json = json.loads(result_clean)
                
                for flag in flags_json.get("flags", []):
                    violation = Violation(
                        regulation=flag.get("regulation", "Unknown"),
                        severity=flag.get("severity", "Minor"),
                        quote=flag.get("quote", "")[:500],
                        explanation=flag.get("explanation", ""),
                        remediation=flag.get("remediation", ""),
                        risk=flag.get("risk", ""),
                        penalties=flag.get("penalties", ""),
                        page=page_number,
                        chunk_id=chunk_id,
                        confidence=flag.get("confidence", 0.9)
                    )
                    violations.append(violation)
                    
            except json.JSONDecodeError as e:
                st.warning(f"⚠️ JSON parsing error for page {page_number}: {str(e)}")
                
        return violations

    def analyze_document(self, pages_text: List[Dict]) -> tuple[List[Violation], int, Dict]:
        """Enhanced document analysis with deduplication and scoring"""
        all_violations = []
        violation_hashes = set()
        stats = {"chunks_processed": 0, "pages_with_text": 0, "api_calls": 0}
        
        total_pages = len([p for p in pages_text if p["text"].strip()])
        
        progress_container = st.container()
        with progress_container:
            st.markdown('<div class="progress-container">', unsafe_allow_html=True)
            st.markdown("### 🧠 Analyzing Document for Regulatory Violations")
            progress_bar = st.progress(0)
            progress_text = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
        
        for page_data in pages_text:
            if not page_data["text"].strip():
                continue
                
            stats["pages_with_text"] += 1
            progress_text.text(f"Analyzing page {page_data['page']} of {total_pages}...")
            
            chunks = self.text_splitter.split_text(page_data["text"])
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"p{page_data['page']}_c{i+1}"
                stats["chunks_processed"] += 1
                
                violations = self.analyze_chunk(chunk, page_data["page"], chunk_id)
                stats["api_calls"] += 1
                
                for violation in violations:
                    violation_hash = self.create_violation_hash(
                        violation.regulation, 
                        violation.quote, 
                        violation.severity
                    )
                    
                    if violation_hash not in violation_hashes:
                        violation_hashes.add(violation_hash)
                        all_violations.append(violation)
            
            progress = stats["pages_with_text"] / total_pages
            progress_bar.progress(progress)
            time.sleep(0.5)
        
        progress_bar.empty()
        progress_text.empty()
        progress_container.empty()
        
        score = self.calculate_compliance_score(all_violations)
        return all_violations, score, stats

    def calculate_compliance_score(self, violations: List[Violation]) -> int:
        """Calculate compliance score with weighted penalties"""
        base_score = 100
        penalty_weights = {
            "Critical": 40,
            "Major": 20, 
            "Minor": 5
        }
        
        total_penalty = 0
        for violation in violations:
            penalty = penalty_weights.get(violation.severity, 5)
            weighted_penalty = penalty * violation.confidence
            total_penalty += weighted_penalty
        
        final_score = max(0, int(base_score - total_penalty))
        return final_score

    def generate_executive_summary(self, pages_text: List[Dict], violations: List[Violation]) -> str:
        """Generate comprehensive executive summary"""
        full_text = " ".join([p["text"] for p in pages_text if p["text"]])[:3000]
        
        violation_summary = []
        severity_counts = Counter([v.severity for v in violations])
        
        for severity in ["Critical", "Major", "Minor"]:
            if severity_counts[severity] > 0:
                violation_summary.append(f"{severity_counts[severity]} {severity.lower()} violations")
        
        violations_text = ", ".join(violation_summary) if violation_summary else "No violations detected"
        
        prompt = f"""
Generate a concise executive summary (150 words max) for this compliance analysis:

DOCUMENT CONTENT: {full_text}

VIOLATIONS FOUND: {violations_text}

TOP VIOLATIONS:
{chr(10).join([f"- Page {v.page}: {v.regulation} - {v.explanation}" for v in violations[:3]])}

Provide a professional summary covering:
1. Document type and purpose
2. Overall compliance status  
3. Key risk areas identified
4. Recommended priority actions
"""
        
        summary = self.call_openrouter(prompt, max_tokens=300)
        return summary or "Unable to generate executive summary due to API limitations."

def display_pdf_viewer(uploaded_file):
    """Display PDF in an iframe for side-by-side viewing"""
    try:
        # Convert PDF to base64 for embedding
        pdf_bytes = uploaded_file.read()
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        
        # Reset file pointer
        uploaded_file.seek(0)
        
        pdf_display = f"""
        <div class="pdf-container">
            <embed src="data:application/pdf;base64,{base64_pdf}" 
                   width="100%" height="100%" type="application/pdf">
        </div>
        """
        
        st.markdown(pdf_display, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")
        st.info("PDF preview not available, but analysis will proceed normally.")

def display_json_viewer(data, title="Analysis Results"):
    """Display JSON data with syntax highlighting"""
    st.markdown(f'<div class="section-header"><h4>{title}</h4></div>', unsafe_allow_html=True)
    
    # Pretty print JSON
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    
    # Create expandable JSON viewer
    with st.expander("📋 View Raw JSON Data", expanded=False):
        st.markdown(f'<div class="json-viewer"><pre>{json_str}</pre></div>', unsafe_allow_html=True)

def display_violation_segments(violations: List[Violation]):
    """Display violations in organized segments"""
    if not violations:
        st.success("🎉 No regulatory violations detected!")
        return
    
    # Group violations by severity
    severity_groups = {
        "Critical": [v for v in violations if v.severity == "Critical"],
        "Major": [v for v in violations if v.severity == "Major"],
        "Minor": [v for v in violations if v.severity == "Minor"]
    }
    
    # Display each severity group
    for severity, severity_violations in severity_groups.items():
        if not severity_violations:
            continue
            
        severity_icon = {"Critical": "🔴", "Major": "🟡", "Minor": "🔵"}[severity]
        severity_color = {"Critical": "critical", "Major": "major", "Minor": "minor"}[severity]
        
        st.markdown(f'<div class="section-header"><h4>{severity_icon} {severity} Violations ({len(severity_violations)})</h4></div>', 
                   unsafe_allow_html=True)
        
        for i, violation in enumerate(severity_violations):
            with st.container():
                st.markdown(f'<div class="violation-card {severity_color}">', unsafe_allow_html=True)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**📍 Page {violation.page} - {violation.regulation}**")
                    
                with col2:
                    if violation.confidence < 1.0:
                        st.caption(f"Confidence: {violation.confidence:.1%}")
                
                # Problematic text
                st.markdown("**🎯 Problematic Text:**")
                st.code(violation.quote, language=None)
                
                # Create tabs for detailed information
                tab1, tab2, tab3, tab4 = st.tabs(["📝 Explanation", "🔧 Remediation", "⚠️ Risk", "💰 Penalties"])
                
                with tab1:
                    st.write(violation.explanation)
                
                with tab2:
                    st.write(violation.remediation)
                
                with tab3:
                    st.write(violation.risk)
                
                with tab4:
                    st.write(violation.penalties)
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    return ComplianceAnalyzer()

analyzer = get_analyzer()

# Main UI
st.markdown("""
<div class="main-header">
    <h1>⚖️ Compliance Document Analyzer Pro</h1>
    <p style="font-size: 1.2em; margin: 0;">Federal & State Regulatory Compliance Validation System</p>
    <p style="font-size: 1em; margin: 0.5rem 0 0 0;">Advanced AI-Powered Risk Assessment with Side-by-Side Document Viewing</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("📋 Configuration Panel")
    
    # OCR Status
    if ocr is None:
        st.error("⚠️ OCR functionality not available")
        st.caption("Text-based PDFs only")
    else:
        st.success("✅ OCR functionality loaded")
    
    # Custom rules upload
    st.subheader("📚 Custom Rules")
    rules_file = st.file_uploader("Upload Custom Rules (JSON)", type="json", key="rules")
    if rules_file:
        try:
            new_rules = json.load(rules_file)
            analyzer.CUSTOM_RULES = new_rules
            st.success("✅ Custom rules updated")
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON format")
    
    st.subheader("🎯 Regulatory Coverage")
    with st.expander("View Covered Regulations"):
        regulations = [
            "FTC Section 5 (UDAAP)",
            "Truth in Lending (Reg Z)",
            "Truth in Savings (Reg DD)", 
            "Fair Housing Act",
            "ECOA (Fair Lending)",
            "GLBA (Privacy)",
            "CCPA/CPRA",
            "TCPA",
            "CAN-SPAM",
            "FDCPA"
        ]
        for reg in regulations:
            st.write(f"• {reg}")
    
    st.subheader("📊 Analysis Settings")
    show_json = st.checkbox("Show JSON Analysis", value=True)
    show_confidence = st.checkbox("Show Confidence Scores", value=True)

# Main content area
uploaded_file = st.file_uploader(
    "📄 Upload PDF Document for Compliance Analysis", 
    type="pdf",
    help="Upload contracts, marketing materials, disclosures, or other business documents"
)

if uploaded_file:
    st.success(f"📁 File loaded: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
    
    # Create main layout with tabs
    main_tabs = st.tabs(["🔍 Analysis", "📄 Document View", "📊 Results Dashboard"])
    
    with main_tabs[1]:  # Document View Tab
        st.markdown('<div class="section-header"><h3>📄 Document Preview</h3></div>', unsafe_allow_html=True)
        display_pdf_viewer(uploaded_file)
    
    with main_tabs[0]:  # Analysis Tab
        if st.button("🚀 Start Compliance Analysis", type="primary", use_container_width=True):
            start_time = time.time()
            
            # Extract text
            st.markdown('<div class="section-header"><h3>📖 Text Extraction</h3></div>', unsafe_allow_html=True)
            pages_text = analyzer.extract_pdf_text(uploaded_file)
            
            if not pages_text or not any(p["text"].strip() for p in pages_text):
                st.error("❌ No readable text found in PDF. Please ensure the document contains text or is scannable.")
                st.stop()
            
            # Analyze document
            violations, score, stats = analyzer.analyze_document(pages_text)
            
            # Generate summary
            st.markdown('<div class="progress-container">📋 Generating executive summary...</div>', unsafe_allow_html=True)
            summary = analyzer.generate_executive_summary(pages_text, violations)
            
            processing_time = time.time() - start_time
            
            # Store results in session state
            st.session_state.analysis_results = {
                'violations': violations,
                'score': score,
                'stats': stats,
                'summary': summary,
                'processing_time': processing_time,
                'pages_text': pages_text
            }
            
            st.success(f"✅ Analysis completed in {processing_time:.1f} seconds")
            st.rerun()
    
    # Display results if available
    if 'analysis_results' in st.session_state:
        results = st.session_state.analysis_results
        violations = results['violations']
        score = results['score']
        stats = results['stats']
        summary = results['summary']
        processing_time = results['processing_time']
        pages_text = results['pages_text']
        
        with main_tabs[2]:  # Results Dashboard Tab
            st.markdown('<div class="section-header"><h3>📊 Analysis Dashboard</h3></div>', unsafe_allow_html=True)
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                score_color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h2 style="margin: 0; color: {'#28a745' if score >= 80 else '#ffc107' if score >= 60 else '#dc3545'}">
                        {score_color} {score}%
                    </h2>
                    <p style="margin: 0.5rem 0 0 0; color: #666;">Compliance Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                violation_color = "#dc3545" if len(violations) > 5 else "#ffc107" if len(violations) > 0 else "#28a745"
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h2 style="margin: 0; color: {violation_color}">
                        {len(violations)}
                    </h2>
                    <p style="margin: 0.5rem 0 0 0; color: #666;">Total Violations</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h2 style="margin: 0; color: #007bff">
                        {stats['pages_with_text']}
                    </h2>
                    <p style="margin: 0.5rem 0 0 0; color: #666;">Pages Analyzed</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h2 style="margin: 0; color: #6f42c1">
                        {processing_time:.1f}s
                    </h2>
                    <p style="margin: 0.5rem 0 0 0; color: #666;">Processing Time</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Executive Summary
            st.markdown('<div class="section-header"><h3>📋 Executive Summary</h3></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #007bff;">
                {summary}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Main Analysis Results (always visible when results exist)
        st.markdown('<div class="section-header"><h2>⚠️ Detailed Violation Analysis</h2></div>', unsafe_allow_html=True)
        
        # Create side-by-side layout for violations and document
        analysis_col, document_col = st.columns([3, 2])
        
        with analysis_col:
            st.markdown("### 📝 Violations by Severity")
            display_violation_segments(violations)
            
            # JSON viewer (if enabled)
            if show_json and violations:
                st.markdown("<br><br>", unsafe_allow_html=True)
                
                # Prepare JSON data for display
                json_data = {
                    "document_analysis": {
                        "document_name": uploaded_file.name,
                        "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "compliance_score": score,
                        "processing_stats": stats,
                        "executive_summary": summary
                    },
                    "violations": [violation.to_dict() for violation in violations]
                }
                
                display_json_viewer(json_data, "📄 Complete Analysis JSON")
        
        with document_col:
            st.markdown("### 📄 Document Reference")
            st.caption("Use this view to cross-reference violations with the source document")
            
            # Display PDF viewer in the sidebar column
            display_pdf_viewer(uploaded_file)
            
            # Violation summary card
            st.markdown("<br>", unsafe_allow_html=True)
            
            severity_counts = Counter([v.severity for v in violations])
            
            summary_html = f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="margin-top: 0; color: #333;">📊 Violation Summary</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                    <div style="text-align: center; padding: 0.5rem; background: #fff5f5; border-radius: 4px;">
                        <strong style="color: #dc3545;">{severity_counts.get('Critical', 0)}</strong><br>
                        <small>Critical</small>
                    </div>
                    <div style="text-align: center; padding: 0.5rem; background: #fffbf0; border-radius: 4px;">
                        <strong style="color: #ffc107;">{severity_counts.get('Major', 0)}</strong><br>
                        <small>Major</small>
                    </div>
                    <div style="text-align: center; padding: 0.5rem; background: #f0f9ff; border-radius: 4px;">
                        <strong style="color: #17a2b8;">{severity_counts.get('Minor', 0)}</strong><br>
                        <small>Minor</small>
                    </div>
                    <div style="text-align: center; padding: 0.5rem; background: #f8f9fa; border-radius: 4px;">
                        <strong style="color: #28a745;">{len(violations)}</strong><br>
                        <small>Total</small>
                    </div>
                </div>
            </div>
            """
            
            st.markdown(summary_html, unsafe_allow_html=True)
        
        # Export section
        st.markdown('<div class="section-header"><h3>💾 Export Analysis Results</h3></div>', unsafe_allow_html=True)
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        # Prepare comprehensive report data
        comprehensive_report = {
            "metadata": {
                "document_name": uploaded_file.name,
                "file_size": uploaded_file.size,
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processing_time_seconds": processing_time,
                "analyzer_version": "2.0.0"
            },
            "compliance_assessment": {
                "overall_score": score,
                "score_interpretation": (
                    "Excellent" if score >= 90 else
                    "Good" if score >= 80 else
                    "Fair" if score >= 70 else
                    "Poor" if score >= 50 else
                    "Critical"
                ),
                "total_violations": len(violations),
                "severity_breakdown": dict(severity_counts)
            },
            "processing_statistics": stats,
            "executive_summary": summary,
            "detailed_violations": [violation.to_dict() for violation in violations],
            "document_structure": {
                "total_pages": len(pages_text),
                "pages_with_content": stats['pages_with_text'],
                "chunks_analyzed": stats['chunks_processed']
            }
        }
        
        with export_col1:
            st.download_button(
                "📥 Download Complete JSON Report",
                data=json.dumps(comprehensive_report, indent=2, ensure_ascii=False),
                file_name=f"compliance_analysis_{uploaded_file.name}_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with export_col2:
            # Create a summary CSV
            if violations:
                csv_data = []
                for v in violations:
                    csv_data.append({
                        "Page": v.page,
                        "Severity": v.severity,
                        "Regulation": v.regulation,
                        "Quote": v.quote[:100] + "..." if len(v.quote) > 100 else v.quote,
                        "Risk": v.risk[:100] + "..." if len(v.risk) > 100 else v.risk,
                        "Confidence": f"{v.confidence:.1%}"
                    })
                
                import pandas as pd
                df = pd.DataFrame(csv_data)
                csv_string = df.to_csv(index=False)
                
                st.download_button(
                    "📊 Download CSV Summary",
                    data=csv_string,
                    file_name=f"violations_summary_{uploaded_file.name}_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with export_col3:
            # Generate a simple text report
            text_report = f"""COMPLIANCE ANALYSIS REPORT
=============================

Document: {uploaded_file.name}
Analyzed: {time.strftime("%Y-%m-%d %H:%M:%S")}
Compliance Score: {score}%

EXECUTIVE SUMMARY:
{summary}

VIOLATIONS SUMMARY:
- Critical: {severity_counts.get('Critical', 0)}
- Major: {severity_counts.get('Major', 0)}
- Minor: {severity_counts.get('Minor', 0)}
- Total: {len(violations)}

DETAILED VIOLATIONS:
"""
            
            for i, v in enumerate(violations, 1):
                text_report += f"""
{i}. Page {v.page} - {v.severity}
   Regulation: {v.regulation}
   Quote: "{v.quote}"
   Issue: {v.explanation}
   Fix: {v.remediation}
   ---
"""
            
            st.download_button(
                "📄 Download Text Report",
                data=text_report,
                file_name=f"compliance_report_{uploaded_file.name}_{int(time.time())}.txt",
                mime="text/plain",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 8px; margin-top: 2rem;">
    <h4 style="margin: 0; color: #333;">Compliance Document Analyzer Pro v2.0</h4>
    <p style="margin: 0.5rem 0; color: #666;">
        Advanced AI-Powered Regulatory Risk Assessment | Built with Streamlit & OpenRouter API
    </p>
    <p style="margin: 0; font-size: 0.9em; color: #999;">
        ⚠️ <strong>Disclaimer:</strong> This tool provides guidance only. Always consult qualified legal counsel for compliance decisions.
    </p>
</div>
""", unsafe_allow_html=True)