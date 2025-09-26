import streamlit as st
import fitz  # PyMuPDF
import paddleocr
import requests
import json
import time
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from collections import defaultdict, Counter
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import hashlib

# OpenRouter API configuration
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = <api_key_placeholder>  # Replace with your actual API key
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}
MODEL = "x-ai/grok-4-fast:free"

# PaddleOCR setup with compatible settings
@st.cache_resource
def load_ocr():
    try:
        # Try with minimal parameters first
        return paddleocr.PaddleOCR(use_angle_cls=True, lang="en")
    except Exception as e:
        st.error(f"Failed to initialize PaddleOCR: {e}")
        # Fallback to basic OCR without advanced features
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
            st.info("📋 Using enhanced default compliance rules")

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
            
            progress_bar = st.progress(0)
            total_pages = len(pdf)
            
            for page_num in range(total_pages):
                page = pdf[page_num]
                
                # Try text extraction first
                text = page.get_text("text").strip()
                
                # Enhanced OCR fallback for scanned pages
                if len(text) < 50 and ocr is not None:  # Only use OCR if available
                    try:
                        # Higher resolution for better OCR
                        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                        img_data = pix.tobytes("png")
                        
                        ocr_result = ocr.ocr(img_data, cls=True)
                        if ocr_result and ocr_result[0]:
                            ocr_texts = []
                            for line in ocr_result[0]:
                                if len(line) > 1 and line[1][1] > 0.7:  # Confidence threshold
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
                        "temperature": 0.1,  # Lower temperature for consistency
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
                # Clean and parse JSON response
                result_clean = re.sub(r'^```json\s*', '', result.strip())
                result_clean = re.sub(r'\s*```$', '', result_clean)
                
                flags_json = json.loads(result_clean)
                
                for flag in flags_json.get("flags", []):
                    violation = Violation(
                        regulation=flag.get("regulation", "Unknown"),
                        severity=flag.get("severity", "Minor"),
                        quote=flag.get("quote", "")[:500],  # Limit quote length
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
        
        # Progress tracking
        total_pages = len([p for p in pages_text if p["text"].strip()])
        progress_bar = st.progress(0)
        
        for page_data in pages_text:
            if not page_data["text"].strip():
                continue
                
            stats["pages_with_text"] += 1
            
            # Create chunks for this page
            chunks = self.text_splitter.split_text(page_data["text"])
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"p{page_data['page']}_c{i+1}"
                stats["chunks_processed"] += 1
                
                # Analyze chunk
                violations = self.analyze_chunk(chunk, page_data["page"], chunk_id)
                stats["api_calls"] += 1
                
                # Deduplicate violations
                for violation in violations:
                    violation_hash = self.create_violation_hash(
                        violation.regulation, 
                        violation.quote, 
                        violation.severity
                    )
                    
                    if violation_hash not in violation_hashes:
                        violation_hashes.add(violation_hash)
                        all_violations.append(violation)
            
            # Update progress
            progress = stats["pages_with_text"] / total_pages
            progress_bar.progress(progress)
            
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
        
        progress_bar.empty()
        
        # Calculate compliance score
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
            # Apply confidence weighting
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

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    return ComplianceAnalyzer()

analyzer = get_analyzer()

# Enhanced Streamlit UI
st.set_page_config(
    page_title="Compliance Document Analyzer",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Advanced Compliance Document Analyzer")
st.markdown("**Federal & State Regulatory Compliance Validation System**")

# Check OCR availability
if ocr is None:
    st.warning("⚠️ OCR functionality not available. The system will work with text-based PDFs only.")
else:
    st.success("✅ OCR functionality loaded successfully")

# Sidebar for configuration
with st.sidebar:
    st.header("📋 Configuration")
    
    # Custom rules upload
    rules_file = st.file_uploader("Upload Custom Rules (JSON)", type="json", key="rules")
    if rules_file:
        try:
            new_rules = json.load(rules_file)
            analyzer.CUSTOM_RULES = new_rules
            st.success("✅ Custom rules updated")
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON format")
    
    st.subheader("🎯 Analysis Scope")
    st.write("**Covered Regulations:**")
    regulations = [
        "FTC Section 5 (UDAAP)",
        "Truth in Lending (Reg Z)",
        "Truth in Savings (Reg DD)", 
        "Fair Housing Act",
        "ECOA (Fair Lending)",
        "GLBA (Privacy)",
        "CCPA/CPRA",
        "TCPA",
        "CAN-SPAM"
    ]
    for reg in regulations:
        st.write(f"• {reg}")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📄 Upload PDF Document", 
        type="pdf",
        help="Upload marketing materials, contracts, disclosures, or other documents for compliance analysis"
    )

with col2:
    if uploaded_file:
        st.success(f"📁 File loaded: {uploaded_file.name}")
        st.info(f"Size: {uploaded_file.size:,} bytes")

if uploaded_file and st.button("🔍 Analyze Document", type="primary"):
    start_time = time.time()
    
    with st.spinner("🔄 Processing document..."):
        # Extract text
        st.info("📖 Extracting text from PDF...")
        pages_text = analyzer.extract_pdf_text(uploaded_file)
        
        if not pages_text or not any(p["text"].strip() for p in pages_text):
            st.error("❌ No readable text found in PDF")
            st.stop()
        
        # Analyze document
        st.info("🧠 Analyzing for regulatory violations...")
        violations, score, stats = analyzer.analyze_document(pages_text)
        
        # Generate summary
        st.info("📋 Generating executive summary...")
        summary = analyzer.generate_executive_summary(pages_text, violations)
    
    processing_time = time.time() - start_time
    
    # Results display
    st.success(f"✅ Analysis completed in {processing_time:.1f} seconds")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score_color = "normal"
        if score >= 80:
            score_color = "normal"
        elif score >= 60:
            score_color = "warning" 
        else:
            score_color = "error"
            
        st.metric("Compliance Score", f"{score}%", delta=f"{score-100}%")
    
    with col2:
        st.metric("Total Violations", len(violations))
    
    with col3:
        st.metric("Pages Analyzed", stats["pages_with_text"])
    
    with col4:
        st.metric("API Calls Made", stats["api_calls"])
    
    # Executive Summary
    st.subheader("📊 Executive Summary")
    st.write(summary)
    
    # Detailed violations
    if violations:
        st.subheader("⚠️ Regulatory Violations Found")
        
        # Violation severity tabs
        severity_tabs = st.tabs(["🔴 Critical", "🟡 Major", "🔵 Minor", "📋 All"])
        
        for i, severity in enumerate(["Critical", "Major", "Minor", "All"]):
            with severity_tabs[i]:
                if severity == "All":
                    filtered_violations = violations
                else:
                    filtered_violations = [v for v in violations if v.severity == severity]
                
                if filtered_violations:
                    for j, violation in enumerate(filtered_violations):
                        with st.expander(f"Page {violation.page}: {violation.regulation} ({violation.severity})"):
                            st.markdown(f"**📍 Problematic Text:**")
                            st.code(violation.quote, language=None)
                            
                            st.markdown(f"**📝 Explanation:**")
                            st.write(violation.explanation)
                            
                            st.markdown(f"**🔧 Remediation:**")
                            st.write(violation.remediation)
                            
                            st.markdown(f"**⚠️ Risk Assessment:**")
                            st.write(violation.risk)
                            
                            st.markdown(f"**💰 Potential Penalties:**")
                            st.write(violation.penalties)
                            
                            if violation.confidence < 1.0:
                                st.caption(f"Confidence: {violation.confidence:.1%}")
                else:
                    st.info(f"No {severity.lower()} violations found.")
    else:
        st.success("🎉 No regulatory violations detected!")
    
    # Download report
    st.subheader("💾 Export Report")
    
    report_data = {
        "document_name": uploaded_file.name,
        "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "compliance_score": score,
        "processing_stats": stats,
        "violations": [
            {
                "page": v.page,
                "regulation": v.regulation,
                "severity": v.severity,
                "quote": v.quote,
                "explanation": v.explanation,
                "remediation": v.remediation,
                "risk": v.risk,
                "penalties": v.penalties,
                "confidence": v.confidence
            }
            for v in violations
        ],
        "executive_summary": summary
    }
    
    st.download_button(
        "📥 Download JSON Report",
        data=json.dumps(report_data, indent=2),
        file_name=f"compliance_report_{uploaded_file.name}_{int(time.time())}.json",
        mime="application/json"
    )

# Footer
st.markdown("---")
st.markdown(
    "**Compliance Document Analyzer** | Built with Streamlit | "
    "⚠️ *This tool provides guidance only - consult legal counsel for compliance decisions*"
)