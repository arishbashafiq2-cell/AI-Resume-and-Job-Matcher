import io
import os
import re
import json
import hashlib
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import fitz
from docx import Document
from groq import Groq
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from xml.sax.saxutils import escape


# ============================================================
# CareerSync AI | Single-file Streamlit application
# ============================================================
st.set_page_config(
    page_title="CareerSync AI | Career Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "CareerSync AI"
DEFAULT_MODEL = "openai/gpt-oss-20b"
MAX_UPLOAD_MB = 10
MAX_TEXT_CHARS = 24000

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --bg:#080D19; --panel:#111A2B; --line:rgba(148,163,184,.16); --muted:#91A0B8; --purple:#9B7BFF; --cyan:#43D9E8; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(ellipse at 12% 0%, rgba(105,76,220,.16), transparent 32%), radial-gradient(ellipse at 90% 8%, rgba(29,179,204,.10), transparent 30%), #080D19; color:#EDF2FF; }
[data-testid="stHeader"] { background:rgba(8,13,25,.82); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#10182A 0%,#0B1220 100%); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:#E7ECF8; }
h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.035em; }
h1 { font-size:2.45rem !important; }
.block-container { padding-top:2rem; padding-bottom:3rem; max-width:1500px; }
.hero { padding:30px 32px; border:1px solid rgba(167,139,250,.28); border-radius:24px; background:linear-gradient(115deg,rgba(100,75,190,.22),rgba(25,150,180,.10)),rgba(17,26,43,.88); box-shadow:0 18px 60px rgba(0,0,0,.22); position:relative; overflow:hidden; }
.hero:after { content:""; position:absolute; width:260px;height:260px;right:-80px;top:-130px;border-radius:50%;background:rgba(117,91,255,.16);filter:blur(8px); }
.eyebrow { color:#AFA0FF; text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; }
.hero h1 { margin:.45rem 0 .55rem 0; }
.hero p { color:#AAB7CD; max-width:760px; margin:0; line-height:1.65; }
.metric-card,.panel-card { border:1px solid var(--line); background:linear-gradient(145deg,rgba(22,33,54,.94),rgba(13,21,36,.94)); border-radius:18px; padding:19px 20px; min-height:112px; box-shadow:0 10px 35px rgba(0,0,0,.12); }
.metric-label { color:#93A3BC; font-size:.82rem; margin-bottom:9px; }
.metric-value { color:#F4F6FF; font-family:'Space Grotesk'; font-size:1.85rem; font-weight:700; }
.metric-note { color:#8C9BB2; font-size:.76rem; margin-top:5px; }
.section-kicker { color:#A99AFF; font-size:.75rem; letter-spacing:.14em; font-weight:700; text-transform:uppercase; margin-bottom:5px; }
.small-muted { color:#91A0B8; font-size:.86rem; }
.pill { display:inline-block; padding:5px 10px; border:1px solid rgba(67,217,232,.25); color:#75E4EC; border-radius:30px; font-size:.74rem; margin:3px 5px 3px 0; background:rgba(67,217,232,.06); }
.good { color:#69DDB4; } .warn { color:#F7C66B; } .bad { color:#FF8B9A; }
div.stButton > button, div.stDownloadButton > button { border-radius:11px; border:1px solid rgba(167,139,250,.42); background:linear-gradient(110deg,#7256D9,#477BCB); color:white; font-weight:700; min-height:2.65rem; transition:all .2s ease; }
div.stButton > button:hover, div.stDownloadButton > button:hover { border-color:#70E2EB; box-shadow:0 0 22px rgba(91,132,255,.25); transform:translateY(-1px); }
.stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"] { background:#0D1728 !important; border-color:rgba(148,163,184,.23) !important; color:#EDF2FF !important; border-radius:10px !important; }
[data-testid="stFileUploader"] { background:rgba(15,25,42,.7); border:1px dashed rgba(155,123,255,.45); border-radius:16px; padding:12px; }
[data-testid="stMetric"] { background:rgba(17,26,43,.8); border:1px solid var(--line); padding:14px; border-radius:14px; }
hr { border-color:var(--line); }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { background:rgba(18,29,47,.8); border-radius:10px 10px 0 0; padding:10px 16px; }
div[data-testid="stExpander"] { border:1px solid var(--line); border-radius:13px; background:rgba(15,24,40,.55); }
footer {visibility:hidden;}
@media(max-width:768px) { .hero{padding:22px 20px}.hero h1{font-size:1.8rem !important}.block-container{padding-left:1rem;padding-right:1rem} }
</style>
""", unsafe_allow_html=True)


# ---------- Session state ----------
DEFAULT_STATE = {
    "page": "Dashboard",
    "resume_text": "",
    "resume_name": "Pasted resume",
    "job_text": "",
    "job_title": "",
    "company": "",
    "resume_profile": None,
    "job_profile": None,
    "analysis": None,
    "analysis_hash": None,
    "history": [],
    "improved": None,
    "roadmap": None,
    "keywords": None,
    "chat": [],
    "last_error": None,
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------- Utilities ----------
def get_config():
    try:
        api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
        model = st.secrets.get("GROQ_MODEL", os.getenv("GROQ_MODEL", DEFAULT_MODEL))
    except Exception:
        api_key = os.getenv("GROQ_API_KEY", "")
        model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    return str(api_key).strip(), str(model).strip()


def get_client():
    key, model = get_config()
    if not key:
        raise ValueError("Groq API key is missing. Add GROQ_API_KEY in Streamlit Cloud → Settings → Secrets.")
    if not model:
        raise ValueError("GROQ_MODEL is empty. Add a supported model name to Streamlit Secrets.")
    return Groq(api_key=key, timeout=60.0, max_retries=1), model


def groq_text(system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 3500) -> str:
    client, model = get_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("The selected Groq model returned an empty response.")
        return content.strip()
    except Exception as exc:
        message = str(exc)
        low = message.lower()
        if "model_decommissioned" in low or "model_not_found" in low or "does not exist" in low or "unknown model" in low:
            raise RuntimeError(f"The configured Groq model '{model}' is unavailable or not enabled. Update GROQ_MODEL in Streamlit Secrets. No model fallback was used.") from exc
        if "401" in message or "authentication" in low or "invalid_api_key" in low:
            raise RuntimeError("Groq authentication failed. Check your GROQ_API_KEY in Streamlit Secrets.") from exc
        if "429" in message or "rate_limit" in low:
            raise RuntimeError("Groq rate limit reached. Please wait briefly and try again.") from exc
        raise RuntimeError(f"Groq request failed: {message[:500]}") from exc


def extract_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        for page in pdf:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts).strip()


def extract_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    chunks = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            chunks.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(chunks).strip()


def safe_json(raw: str) -> Dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise ValueError("AI returned an invalid structured response. Please retry.")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError("AI returned malformed JSON. Please retry.") from exc
    if not isinstance(data, dict):
        raise ValueError("AI response was not a JSON object.")
    return data


def ai_json(system: str, prompt: str, max_tokens: int = 4000) -> Dict[str, Any]:
    raw = groq_text(system + "\nReturn only valid JSON. Do not wrap it in markdown.", prompt, max_tokens=max_tokens)
    return safe_json(raw)


def clean_list(value):
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [x.strip() for x in re.split(r"[,;\n]", value) if x.strip()]
    return []


def clamp_score(value):
    try:
        return max(0, min(100, int(float(value))))
    except (TypeError, ValueError):
        return 0


def text_hash(*parts):
    return hashlib.sha256("||".join(parts).encode("utf-8", errors="ignore")).hexdigest()


def metric_card(label, value, note=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{escape(str(label))}</div><div class="metric-value">{escape(str(value))}</div><div class="metric-note">{escape(str(note))}</div></div>', unsafe_allow_html=True)


def render_hero(title, subtitle, eyebrow="CAREER INTELLIGENCE"):
    st.markdown(f'<div class="hero"><div class="eyebrow">{escape(eyebrow)}</div><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>', unsafe_allow_html=True)


def render_tags(items, empty="No items identified yet."):
    items = clean_list(items)
    if not items:
        st.caption(empty)
        return
    st.markdown("".join(f'<span class="pill">{escape(x)}</span>' for x in items), unsafe_allow_html=True)


def require_context():
    if not st.session_state.resume_text.strip():
        st.warning("Add a resume first in Resume Analyzer.")
        return False
    if not st.session_state.job_text.strip():
        st.warning("Add a job description first in Job Description Analyzer.")
        return False
    return True


def save_history():
    a = st.session_state.analysis
    if not a:
        return
    item = {
        "resume_name": st.session_state.resume_name,
        "job_title": st.session_state.job_title or "Untitled role",
        "company": st.session_state.company,
        "score": clamp_score(a.get("overall_match_score", 0)),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "matched": clean_list(a.get("matching_skills")),
        "missing": clean_list(a.get("missing_skills")),
        "analysis": a,
    }
    st.session_state.history.insert(0, item)
    st.session_state.history = st.session_state.history[:30]


# ---------- AI feature prompts ----------
RESUME_SYSTEM = """You are a careful professional resume analyst. Extract only facts explicitly supported by the supplied resume. Never infer or invent names, employers, dates, degrees, certifications, metrics, or skills. Return JSON with keys: full_name, email, phone, headline, professional_summary, technical_skills (array), soft_skills (array), experience (array of objects with role, company, dates, details), education (array), certifications (array), projects (array), achievements (array), sections_found (array), extraction_notes (array). Use empty strings or arrays when unavailable."""
JOB_SYSTEM = """You are a precise recruitment requirements analyst. Extract requirements from the job description only. Separate mandatory requirements from preferred qualifications. Return JSON keys: role, seniority, required_skills (array), preferred_skills (array), soft_skills (array), tools_technologies (array), experience_requirement, education_requirement, certifications (array), responsibilities (array), keywords (array), other_requirements (array). Do not invent requirements."""
MATCH_SYSTEM = """You are an evidence-based resume-to-job comparison engine. Evaluate only evidence in the supplied resume and job description. Do not infer a qualification from a job title alone. Scores are analytical estimates, not ATS scores or hiring probabilities. Return JSON keys: overall_match_score (0-100 integer), category_scores (object with technical_skills, soft_skills, experience, education, keywords each 0-100), matching_skills (array), missing_skills (array), partial_skills (array), relevant_experience (array), missing_qualifications (array), strengths (array), weaknesses (array), ats_issues (array), score_explanation (string), recommendations (array). Clearly distinguish mandatory from preferred requirements where possible."""
IMPROVE_SYSTEM = """You are an ethical resume editor. Improve wording without adding facts. Never fabricate numbers, outcomes, responsibilities, employers, qualifications, tools, certifications, or achievements. If measurable impact is missing, provide a placeholder suggestion explicitly marked [add verified metric if available] rather than inventing a number. Return JSON keys: improved_summary (string), bullet_rewrites (array of objects with original, improved, rationale), action_verbs (array), keyword_suggestions (array), weak_phrases (array of objects with phrase, alternative), skills_presentation (string), section_order (array), truthfulness_notes (array)."""
ROADMAP_SYSTEM = """You are a practical career learning coach. Base the roadmap on the actual gap analysis and job description. Distinguish mandatory gaps from preferred gaps. Do not imply that learning a skill guarantees employment. Return JSON keys: roadmap (array of objects with skill, priority [High/Medium/Low], requirement_type [Mandatory/Preferred], why_it_matters, level [Beginner/Intermediate/Advanced], learning_topics (array), practical_project, sequence integer), overall_strategy (string)."""
KEYWORD_SYSTEM = """You are an ATS keyword analysis assistant. Compare exact and semantically equivalent phrases in a resume and job description. Never recommend hidden text or keyword stuffing. Return JSON keys: present_keywords (array), missing_keywords (array), underused_keywords (array), placement_recommendations (array of objects with keyword, section, natural_wording), caution (string)."""
CHAT_SYSTEM = """You are CareerSync AI, a grounded career assistant. Answer using only the resume and job description context supplied. Be candid, constructive, concise, and practical. Do not claim analysis or experience not present in the context. Clearly say when information is missing. Never fabricate resume facts."""


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ✦ CareerSync **AI**")
    st.caption("INTELLIGENT CAREER PLATFORM")
    st.markdown("---")
    pages = [
        ("Dashboard", "⌂"), ("Resume Analyzer", "▤"), ("Job Description", "⌕"),
        ("Job Matching", "◈"), ("Improvement Studio", "✧"), ("Skills Roadmap", "⌁"),
        ("ATS Optimizer", "◎"), ("Career Assistant", "✦"), ("Analysis History", "◷"),
        ("Settings", "⚙"),
    ]
    for page, icon in pages:
        if st.button(f"{icon}  {page}", key=f"nav_{page}", use_container_width=True,
                     type="primary" if st.session_state.page == page else "secondary"):
            st.session_state.page = page
            st.rerun()
    st.markdown("---")
    api_key, model_name = get_config()
    st.markdown("**AI connection**")
    if api_key:
        st.markdown('<span class="good">● API key configured</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="warn">● API key required</span>', unsafe_allow_html=True)
    st.caption(f"Model: {model_name or 'Not configured'}")
    if st.button("✦ Ask Career Assistant", use_container_width=True):
        st.session_state.page = "Career Assistant"
        st.rerun()
    st.markdown("---")
    st.caption("Your work stays in this browser session. No permanent database is configured.")


# ---------- Dashboard ----------
page = st.session_state.page
if page == "Dashboard":
    render_hero("Your next career move starts here.", "Turn your resume and a target job description into clear, evidence-based career intelligence.", "CAREERSYNC AI / OVERVIEW")
    st.write("")
    history = st.session_state.history
    scores = [x["score"] for x in history]
    latest = st.session_state.analysis or {}
    total_skills = len(clean_list(latest.get("matching_skills"))) + len(clean_list(latest.get("missing_skills")))
    cols = st.columns(4)
    with cols[0]: metric_card("Resumes analyzed", len(history), "Analyses in this session")
    with cols[1]: metric_card("Average match", f"{round(sum(scores)/len(scores))}%" if scores else "—", "Across saved analyses")
    with cols[2]: metric_card("Skills identified", total_skills if latest else "—", "Latest job comparison")
    with cols[3]: metric_card("Missing skills", len(clean_list(latest.get("missing_skills"))) if latest else "—", "Latest job comparison")
    st.write("")
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("### Start your analysis")
        st.markdown('<div class="panel-card"><div class="section-kicker">3-step workflow</div><h3>From resume to a stronger application</h3><p class="small-muted">Upload your resume, add a job description, then let the matching engine identify evidence-backed strengths and gaps.</p></div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        with a:
            if st.button("01 · Add resume", use_container_width=True):
                st.session_state.page = "Resume Analyzer"; st.rerun()
        with b:
            if st.button("02 · Add job", use_container_width=True):
                st.session_state.page = "Job Description"; st.rerun()
        with c:
            if st.button("03 · Match now", use_container_width=True):
                st.session_state.page = "Job Matching"; st.rerun()
        st.markdown("### Recent analyses")
        if history:
            for item in history[:4]:
                st.markdown(f'<div class="panel-card" style="margin-bottom:9px;padding:14px 17px"><b>{escape(item["job_title"])}</b> <span style="float:right;color:#75E4EC;font-weight:700">{item["score"]}% match</span><div class="small-muted">{escape(item["resume_name"])} · {escape(item["date"])}</div></div>', unsafe_allow_html=True)
        else:
            st.info("Your completed analyses will appear here.")
    with right:
        st.markdown("### Career snapshot")
        if latest:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=clamp_score(latest.get("overall_match_score", 0)),
                number={"suffix":"%","font":{"color":"#F4F6FF","size":36}},
                gauge={"axis":{"range":[0,100],"tickcolor":"#91A0B8"},
                       "bar":{"color":"#9B7BFF"}, "bgcolor":"rgba(255,255,255,.04)",
                       "steps":[{"range":[0,50],"color":"rgba(255,139,154,.12)"},{"range":[50,75],"color":"rgba(247,198,107,.12)"},{"range":[75,100],"color":"rgba(105,221,180,.12)"}]}))
            fig.update_layout(height=260, margin=dict(l=20,r=20,t=25,b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="#EDF2FF")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Evidence-based compatibility estimate. Not an ATS score or hiring probability.")
        else:
            st.markdown('<div class="panel-card"><h3>✦ Your career intelligence</h3><p class="small-muted">Your first analysis will unlock a match gauge, category insights, skill gaps, and targeted recommendations.</p></div>', unsafe_allow_html=True)
        st.markdown("### What you get")
        for line in ["Skill-by-skill comparison", "Truthful resume improvements", "Prioritized learning roadmap", "ATS keyword placement guidance"]:
            st.markdown(f"✓ {line}")


# ---------- Resume Analyzer ----------
elif page == "Resume Analyzer":
    render_hero("Resume Analyzer", "Upload a PDF or DOCX resume, or paste your content. CareerSync extracts only the information present in your document.", "PROFILE INTELLIGENCE")
    st.write("")
    uploaded = st.file_uploader("Upload your resume", type=["pdf","docx"], help="Maximum file size: 10 MB. Scanned image-only PDFs may require OCR before text can be extracted.")
    if uploaded:
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File exceeds the {MAX_UPLOAD_MB} MB limit.")
        else:
            try:
                raw = uploaded.getvalue()
                extracted = extract_pdf(raw) if uploaded.name.lower().endswith(".pdf") else extract_docx(raw)
                if not extracted.strip():
                    st.error("No selectable text was found. This may be a scanned PDF. Please paste the resume text or upload a text-based PDF/DOCX.")
                else:
                    st.session_state.resume_text = extracted[:MAX_TEXT_CHARS]
                    st.session_state.resume_name = uploaded.name
                    st.session_state.resume_profile = None
                    st.session_state.analysis = None
                    st.success(f"Extracted {len(extracted):,} characters from {uploaded.name}.")
            except Exception as exc:
                st.error(f"Could not read this file: {str(exc)[:350]}")
    manual = st.text_area("Or paste resume text", value=st.session_state.resume_text if not uploaded else st.session_state.resume_text,
                          height=260, placeholder="Paste your professional summary, experience, education, skills, projects and achievements...")
    c1, c2 = st.columns([1,3])
    with c1:
        if st.button("Save resume text", use_container_width=True):
            if not manual.strip():
                st.warning("Please paste resume content first.")
            else:
                st.session_state.resume_text = manual[:MAX_TEXT_CHARS]
                if not uploaded: st.session_state.resume_name = "Pasted resume"
                st.session_state.resume_profile = None
                st.session_state.analysis = None
                st.success("Resume content saved for this session.")
    with c2:
        st.caption(f"{len(manual):,} characters · Maximum {MAX_TEXT_CHARS:,} characters used for AI analysis")
    if st.session_state.resume_text:
        with st.expander("Preview extracted resume", expanded=True):
            st.text(st.session_state.resume_text[:12000])
            if len(st.session_state.resume_text) > 12000: st.caption("Preview truncated; full extracted text remains available to analysis.")
        if st.button("✦ Analyze resume profile", type="primary"):
            with st.spinner("Reading your resume and extracting profile details..."):
                try:
                    profile = ai_json(RESUME_SYSTEM, "RESUME:\n" + st.session_state.resume_text[:MAX_TEXT_CHARS])
                    st.session_state.resume_profile = profile
                    st.success("Resume profile analyzed.")
                except Exception as exc: st.error(str(exc))
    profile = st.session_state.resume_profile
    if profile:
        st.markdown("### Extracted professional profile")
        st.markdown(f"**{profile.get('full_name') or 'Name not provided'}** · {profile.get('headline') or 'Professional headline not identified'}")
        st.caption(profile.get("email") or "Email not identified")
        x, y = st.columns(2)
        with x:
            st.markdown("**Technical skills**"); render_tags(profile.get("technical_skills"))
            st.markdown("**Soft skills**"); render_tags(profile.get("soft_skills"))
            st.markdown("**Certifications**"); render_tags(profile.get("certifications"))
        with y:
            st.markdown("**Education**")
            for e in profile.get("education", []): st.write("•", e)
            st.markdown("**Projects**")
            for e in profile.get("projects", []): st.write("•", e)
            st.markdown("**Achievements**")
            for e in profile.get("achievements", []): st.write("•", e)
        if profile.get("extraction_notes"):
            st.info(" · ".join(clean_list(profile["extraction_notes"])))


# ---------- Job Description ----------
elif page == "Job Description":
    render_hero("Job Description Analyzer", "Define the role you want. The AI will separate mandatory requirements from preferred qualifications.", "ROLE INTELLIGENCE")
    st.write("")
    a,b = st.columns(2)
    with a: title = st.text_input("Job title", value=st.session_state.job_title, placeholder="e.g. Junior Python Developer")
    with b: company = st.text_input("Company (optional)", value=st.session_state.company, placeholder="e.g. Acme Technologies")
    jd = st.text_area("Paste the complete job description", value=st.session_state.job_text, height=300,
                      placeholder="Paste responsibilities, required qualifications, preferred skills, tools and experience...")
    if st.button("Save job description", type="primary"):
        if len(jd.strip()) < 80:
            st.warning("Please provide a more complete job description (at least 80 characters).")
        else:
            changed = jd != st.session_state.job_text or title != st.session_state.job_title
            st.session_state.job_text = jd[:MAX_TEXT_CHARS]
            st.session_state.job_title = title.strip()
            st.session_state.company = company.strip()
            if changed:
                st.session_state.job_profile = None
                st.session_state.analysis = None
            st.success("Job description saved for this session.")
    if st.session_state.job_text:
        st.caption(f"{len(st.session_state.job_text):,} characters saved")
        if st.button("✦ Analyze job requirements"):
            with st.spinner("Extracting role requirements and keywords..."):
                try:
                    profile = ai_json(JOB_SYSTEM, f"JOB TITLE: {st.session_state.job_title}\nCOMPANY: {st.session_state.company}\nJOB DESCRIPTION:\n{st.session_state.job_text[:MAX_TEXT_CHARS]}")
                    st.session_state.job_profile = profile
                    st.success("Job requirements analyzed.")
                except Exception as exc: st.error(str(exc))
    p = st.session_state.job_profile
    if p:
        st.markdown("### Role intelligence")
        st.markdown(f"**{p.get('role') or st.session_state.job_title or 'Role'}** · {p.get('seniority') or 'Seniority not specified'}")
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Mandatory skills**"); render_tags(p.get("required_skills"))
            st.markdown("**Preferred skills**"); render_tags(p.get("preferred_skills"))
            st.markdown("**Tools & technologies**"); render_tags(p.get("tools_technologies"))
        with c2:
            st.markdown("**Soft skills**"); render_tags(p.get("soft_skills"))
            st.markdown("**Important keywords**"); render_tags(p.get("keywords"))
            st.markdown("**Experience / education**")
            st.write(p.get("experience_requirement") or "Not specified")
            st.write(p.get("education_requirement") or "Not specified")
        with st.expander("Responsibilities and other requirements"):
            for item in clean_list(p.get("responsibilities")) + clean_list(p.get("other_requirements")): st.write("•",item)


# ---------- Matching Engine ----------
elif page == "Job Matching":
    render_hero("AI Job Matching Engine", "Compare your resume against a target role with transparent, evidence-based compatibility insights.", "MATCH INTELLIGENCE")
    st.write("")
    if st.session_state.resume_text: st.success(f"Resume ready: {st.session_state.resume_name}")
    else: st.warning("Resume not added yet.")
    if st.session_state.job_text: st.success(f"Job description ready: {st.session_state.job_title or 'Untitled role'}")
    else: st.warning("Job description not added yet.")
    if st.button("✦ Analyze My Match", type="primary", use_container_width=True, disabled=not (st.session_state.resume_text.strip() and st.session_state.job_text.strip())):
        current_hash = text_hash(st.session_state.resume_text, st.session_state.job_text, st.session_state.job_title)
        if st.session_state.analysis_hash == current_hash and st.session_state.analysis:
            st.info("These exact inputs have already been analyzed in this session. Showing the existing result.")
        else:
            prompt = f"""RESUME:\n{st.session_state.resume_text[:MAX_TEXT_CHARS]}\n\nJOB TITLE: {st.session_state.job_title}\nCOMPANY: {st.session_state.company}\nJOB DESCRIPTION:\n{st.session_state.job_text[:MAX_TEXT_CHARS]}"""
            with st.spinner("Comparing qualifications, skills, experience and keywords..."):
                try:
                    result = ai_json(MATCH_SYSTEM, prompt, max_tokens=4500)
                    result["overall_match_score"] = clamp_score(result.get("overall_match_score"))
                    cats = result.get("category_scores", {})
                    result["category_scores"] = {k: clamp_score(cats.get(k,0)) for k in ["technical_skills","soft_skills","experience","education","keywords"]}
                    st.session_state.analysis = result
                    st.session_state.analysis_hash = current_hash
                    save_history()
                    st.session_state.improved = None
                    st.session_state.roadmap = None
                    st.session_state.keywords = None
                    st.success("Your match analysis is ready.")
                except Exception as exc:
                    st.error(str(exc))
    result = st.session_state.analysis
    if result:
        score = clamp_score(result.get("overall_match_score"))
        st.markdown("---")
        st.markdown("### Your compatibility overview")
        left,right = st.columns([1,1.5])
        with left:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=score, number={"suffix":"%","font":{"size":42,"color":"#F5F5FF"}},
                title={"text":"Overall match estimate","font":{"size":15,"color":"#AAB7CD"}},
                gauge={"axis":{"range":[0,100],"tickcolor":"#93A3BC"},"bar":{"color":"#9B7BFF","thickness":.25},
                       "bgcolor":"rgba(255,255,255,.04)","steps":[{"range":[0,50],"color":"rgba(255,139,154,.12)"},{"range":[50,75],"color":"rgba(247,198,107,.12)"},{"range":[75,100],"color":"rgba(105,221,180,.12)"}]}))
            fig.update_layout(height=280, margin=dict(l=18,r=18,t=45,b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="#EDF2FF")
            st.plotly_chart(fig,use_container_width=True)
        with right:
            st.markdown("**Category compatibility**")
            cats = result.get("category_scores",{})
            df = pd.DataFrame({"Category":[x.replace("_"," ").title() for x in cats], "Score":[clamp_score(v) for v in cats.values()]})
            fig2 = px.bar(df,x="Score",y="Category",orientation="h",range_x=[0,100],text="Score")
            fig2.update_traces(marker_color="#8F78F5",texttemplate="%{text}%",textposition="outside")
            fig2.update_layout(height=270,margin=dict(l=5,r=35,t=5,b=5),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#DCE5F5",xaxis_gridcolor="rgba(148,163,184,.12)",yaxis_title="")
            st.plotly_chart(fig2,use_container_width=True)
        st.caption(result.get("score_explanation","Scores are estimates based on the supplied text."))
        tabs = st.tabs(["Skills comparison","Strengths & gaps","ATS considerations","Recommendations"])
        with tabs[0]:
            x,y,z = st.columns(3)
            with x:
                st.markdown("**Matching skills**"); render_tags(result.get("matching_skills"),"No direct matches identified.")
            with y:
                st.markdown("**Partially matching**"); render_tags(result.get("partial_skills"),"No partial matches identified.")
            with z:
                st.markdown("**Missing skills**"); render_tags(result.get("missing_skills"),"No missing skills identified.")
        with tabs[1]:
            x,y = st.columns(2)
            with x:
                st.markdown("**Your strengths**")
                for v in clean_list(result.get("strengths")): st.write("✓",v)
                st.markdown("**Relevant experience**")
                for v in clean_list(result.get("relevant_experience")): st.write("•",v)
            with y:
                st.markdown("**Areas to improve**")
                for v in clean_list(result.get("weaknesses")): st.write("•",v)
                st.markdown("**Missing qualifications**")
                for v in clean_list(result.get("missing_qualifications")): st.write("•",v)
        with tabs[2]:
            for v in clean_list(result.get("ats_issues")): st.warning(v)
            if not clean_list(result.get("ats_issues")): st.success("No specific ATS formatting concerns were identified from the supplied text.")
            st.caption("This is not a proprietary ATS scan and does not predict hiring outcomes.")
        with tabs[3]:
            for i,v in enumerate(clean_list(result.get("recommendations")),1): st.write(f"{i}. {v}")
        report_json = json.dumps(result,indent=2,ensure_ascii=False)
        st.download_button("Download analysis JSON",report_json,file_name="careersync_analysis.json",mime="application/json")
        if st.button("Open Resume Improvement Studio"):
            st.session_state.page="Improvement Studio"; st.rerun()
        st.info("Match score is an analytical estimate based on text evidence, not a guaranteed ATS score or probability of being hired.")


# ---------- Improvement Studio ----------
elif page == "Improvement Studio":
    render_hero("Resume Improvement Studio", "Strengthen your professional language while preserving the truth of your experience.", "RESUME OPTIMIZATION")
    st.write("")
    if not require_context(): st.stop()
    if st.button("✦ Generate resume improvements", type="primary"):
        with st.spinner("Finding stronger, truthful ways to present your experience..."):
            try:
                st.session_state.improved = ai_json(IMPROVE_SYSTEM, f"RESUME:\n{st.session_state.resume_text[:MAX_TEXT_CHARS]}\n\nJOB DESCRIPTION:\n{st.session_state.job_text[:MAX_TEXT_CHARS]}", max_tokens=4500)
                st.success("Improvement suggestions generated.")
            except Exception as exc: st.error(str(exc))
    data = st.session_state.improved
    if data:
        st.markdown("### Professional summary")
        st.markdown('<div class="panel-card">',unsafe_allow_html=True)
        st.write(data.get("improved_summary",""))
        st.markdown("</div>",unsafe_allow_html=True)
        st.code(data.get("improved_summary",""),language="text")
        for i,item in enumerate(data.get("bullet_rewrites",[])):
            with st.expander(f"Experience rewrite {i+1}: {str(item.get('original',''))[:75]}"):
                a,b=st.columns(2)
                with a: st.markdown("**Before**"); st.write(item.get("original",""))
                with b: st.markdown("**Suggested version**"); st.write(item.get("improved","")); st.code(item.get("improved",""),language="text")
                st.caption(item.get("rationale",""))
        a,b=st.columns(2)
        with a:
            st.markdown("**Action verbs**"); render_tags(data.get("action_verbs"))
            st.markdown("**Keyword suggestions**"); render_tags(data.get("keyword_suggestions"))
        with b:
            st.markdown("**Weak phrases to reconsider**")
            for item in data.get("weak_phrases",[]): st.write(f"• {item.get('phrase','')} → {item.get('alternative','')}")
            st.markdown("**Skills presentation**"); st.write(data.get("skills_presentation",""))
        st.markdown("**Suggested section order**")
        st.write(" → ".join(clean_list(data.get("section_order"))))
        if data.get("truthfulness_notes"):
            st.warning("Truthfulness check: " + " · ".join(clean_list(data["truthfulness_notes"])))
        export_text = "PROFESSIONAL SUMMARY\n" + data.get("improved_summary","") + "\n\nEXPERIENCE BULLET SUGGESTIONS\n"
        for item in data.get("bullet_rewrites",[]): export_text += f"\nOriginal: {item.get('original','')}\nSuggested: {item.get('improved','')}\n"
        st.download_button("Download improvement suggestions (.txt)",export_text,file_name="careersync_resume_improvements.txt",mime="text/plain")
        st.caption("Review every suggestion before using it. Only retain wording that accurately reflects your real experience.")


# ---------- Roadmap ----------
elif page == "Skills Roadmap":
    render_hero("Missing Skills & Learning Roadmap", "Turn your skill gaps into a practical learning sequence tied to the target role.", "CAREER DEVELOPMENT")
    st.write("")
    if not require_context(): st.stop()
    if not st.session_state.analysis:
        st.warning("Run Job Matching first so the roadmap can use your actual comparison results.")
    else:
        if st.button("✦ Generate my learning roadmap",type="primary"):
            with st.spinner("Building a prioritized learning plan..."):
                try:
                    prompt=f"RESUME:\n{st.session_state.resume_text[:MAX_TEXT_CHARS]}\nJOB:\n{st.session_state.job_text[:MAX_TEXT_CHARS]}\nMATCH ANALYSIS:\n{json.dumps(st.session_state.analysis,ensure_ascii=False)}"
                    st.session_state.roadmap=ai_json(ROADMAP_SYSTEM,prompt,max_tokens=4000)
                except Exception as exc: st.error(str(exc))
        data=st.session_state.roadmap
        if data:
            st.markdown("### Your learning strategy")
            st.write(data.get("overall_strategy",""))
            for item in sorted(data.get("roadmap",[]),key=lambda x:int(x.get("sequence",99) or 99)):
                priority=item.get("priority","Medium")
                with st.expander(f"{item.get('sequence','•')}. {item.get('skill','Skill')}  ·  {priority} priority  ·  {item.get('requirement_type','Requirement')}",expanded=priority=="High"):
                    st.write(item.get("why_it_matters",""))
                    st.caption(f"Suggested level: {item.get('level','Not specified')} · {item.get('requirement_type','Not specified')}")
                    st.markdown("**Learning topics**")
                    for topic in clean_list(item.get("learning_topics")): st.write("•",topic)
                    st.markdown("**Practical project**")
                    st.write(item.get("practical_project",""))
            st.caption("Priorities reflect the supplied job description and comparison, not a guarantee of employment.")


# ---------- ATS Optimizer ----------
elif page == "ATS Optimizer":
    render_hero("ATS Keyword Optimizer", "Find important role language that is already represented, missing, or could be placed more clearly.", "KEYWORD INTELLIGENCE")
    st.write("")
    if not require_context(): st.stop()
    if st.button("✦ Optimize keyword alignment",type="primary"):
        with st.spinner("Comparing resume language with job requirements..."):
            try:
                st.session_state.keywords=ai_json(KEYWORD_SYSTEM,f"RESUME:\n{st.session_state.resume_text[:MAX_TEXT_CHARS]}\n\nJOB DESCRIPTION:\n{st.session_state.job_text[:MAX_TEXT_CHARS]}",max_tokens=3500)
            except Exception as exc: st.error(str(exc))
    data=st.session_state.keywords
    if data:
        a,b,c=st.columns(3)
        with a: metric_card("Present keywords",len(clean_list(data.get("present_keywords"))),"Already represented")
        with b: metric_card("Missing keywords",len(clean_list(data.get("missing_keywords"))),"Potential content gaps")
        with c: metric_card("Underused keywords",len(clean_list(data.get("underused_keywords"))),"Could be clearer")
        x,y=st.columns(2)
        with x:
            st.markdown("**Already present**"); render_tags(data.get("present_keywords"))
            st.markdown("**Underused**"); render_tags(data.get("underused_keywords"))
        with y:
            st.markdown("**Important missing keywords**"); render_tags(data.get("missing_keywords"))
        st.markdown("### Natural placement recommendations")
        for item in data.get("placement_recommendations",[]):
            st.markdown(f"**{item.get('keyword','')}** · {item.get('section','')}")
            st.write(item.get("natural_wording",""))
        st.info(data.get("caution","Only add keywords that truthfully describe your skills and experience. Never use hidden text or keyword stuffing."))
        st.download_button("Download keyword gap report",json.dumps(data,indent=2),file_name="careersync_keyword_report.json",mime="application/json")


# ---------- Career Assistant ----------
elif page == "Career Assistant":
    render_hero("AI Career Assistant", "Ask focused questions about your resume, the target role, interview preparation, and next steps.", "CAREER COPILOT")
    st.write("")
    if not st.session_state.resume_text and not st.session_state.job_text:
        st.info("Add a resume and/or job description to give the assistant useful context. It can answer general career questions, but will not pretend to know your profile.")
    suggestions=["Am I suitable for this job?","What are my biggest weaknesses?","Which skills should I improve first?","What interview questions should I prepare for?","How can I improve my professional summary?"]
    cols=st.columns(len(suggestions))
    for i,q in enumerate(suggestions):
        with cols[i]:
            if st.button(q,key=f"suggest_{i}",use_container_width=True):
                st.session_state.pending_question=q
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    question=st.chat_input("Ask CareerSync about your career...")
    if st.session_state.get("pending_question"):
        question=st.session_state.pending_question
        st.session_state.pending_question=""
    if question:
        st.session_state.chat.append({"role":"user","content":question})
        with st.chat_message("user"): st.markdown(question)
        context=f"RESUME:\n{st.session_state.resume_text[:12000] or '[Not provided]'}\n\nJOB DESCRIPTION:\n{st.session_state.job_text[:12000] or '[Not provided]'}\n\nMATCH ANALYSIS:\n{json.dumps(st.session_state.analysis,ensure_ascii=False)[:9000] if st.session_state.analysis else '[Not run]'}"
        recent=st.session_state.chat[-10:]
        system=CHAT_SYSTEM+"\n\nCONTEXT:\n"+context
        try:
            client, model=get_client()
            messages=[{"role":"system","content":system}]
            messages.extend(recent)
            with st.chat_message("assistant"):
                with st.spinner("Thinking through your career question..."):
                    response=client.chat.completions.create(model=model,messages=messages,temperature=.35,max_tokens=1400)
                    answer=response.choices[0].message.content or "I couldn't generate a response. Please try again."
                    st.markdown(answer)
            st.session_state.chat.append({"role":"assistant","content":answer})
        except Exception as exc:
            st.error(str(exc))
            st.session_state.chat.pop()
    if st.session_state.chat and st.button("Clear conversation"):
        st.session_state.chat=[]
        st.rerun()


# ---------- Analysis History ----------
elif page == "Analysis History":
    render_hero("Analysis History", "Review analyses created during your current session. History is not permanent storage.", "YOUR WORKSPACE")
    st.write("")
    history=st.session_state.history
    if not history:
        st.info("No analyses saved yet. Complete a job match to create your first history entry.")
    else:
        if st.button("Clear session history",type="secondary"):
            st.session_state.history=[]
            st.session_state.analysis=None
            st.session_state.analysis_hash=None
            st.rerun()
        df=pd.DataFrame([{k:v for k,v in x.items() if k in ["date","resume_name","job_title","company","score"]} for x in history])
        st.dataframe(df,use_container_width=True,hide_index=True)
        for i,item in enumerate(history):
            with st.expander(f"{item['job_title']} · {item['score']}% · {item['date']}"):
                st.write(f"Resume: {item['resume_name']} · Company: {item['company'] or 'Not provided'}")
                st.markdown("**Matched skills**"); render_tags(item["matched"])
                st.markdown("**Missing skills**"); render_tags(item["missing"])
                if st.button("Load this analysis",key=f"load_history_{i}"):
                    st.session_state.analysis=item["analysis"]
                    st.session_state.job_title=item["job_title"]
                    st.session_state.page="Job Matching"
                    st.rerun()
        hist_json=json.dumps(history,indent=2,ensure_ascii=False,default=str)
        st.download_button("Export session history as JSON",hist_json,file_name="careersync_history.json",mime="application/json")


# ---------- PDF report helper ----------
def build_pdf_report(analysis, job_title, resume_name):
    buffer=io.BytesIO()
    doc=SimpleDocTemplate(buffer,pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=18*mm,bottomMargin=18*mm)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Brand",parent=styles["Title"],textColor=colors.HexColor("#5543A8"),fontSize=22,spaceAfter=8))
    styles.add(ParagraphStyle(name="Sub",parent=styles["Normal"],textColor=colors.HexColor("#596579"),fontSize=9,spaceAfter=12))
    styles.add(ParagraphStyle(name="SectionX",parent=styles["Heading2"],textColor=colors.HexColor("#44378C"),fontSize=13,spaceBefore=13,spaceAfter=6))
    styles.add(ParagraphStyle(name="BodyX",parent=styles["BodyText"],fontSize=9,leading=13,spaceAfter=5))
    story=[Paragraph("CareerSync AI",styles["Brand"]),Paragraph("Resume & Job Match Intelligence Report",styles["Sub"]),
           Paragraph(f"<b>Role:</b> {escape(job_title or 'Untitled role')} &nbsp;&nbsp; <b>Resume:</b> {escape(resume_name)}",styles["BodyX"]),
           Paragraph(f"<b>Overall match estimate: {clamp_score(analysis.get('overall_match_score'))}%</b>",styles["Heading1"]),
           Paragraph(escape(analysis.get("score_explanation","")),styles["BodyX"])]
    cats=analysis.get("category_scores",{})
    if cats:
        story += [Paragraph("Category scores",styles["SectionX"]),Table([["Category","Score"]]+[[k.replace("_"," ").title(),f"{clamp_score(v)}%"] for k,v in cats.items()],colWidths=[100*mm,45*mm],style=TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EAE7FA")),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#D6D9E2")),("PADDING",(0,0),(-1,-1),7)]))]
    for heading,key in [("Matching skills","matching_skills"),("Missing skills","missing_skills"),("Partially matching skills","partial_skills"),("Strengths","strengths"),("Weaknesses","weaknesses"),("Missing qualifications","missing_qualifications"),("ATS considerations","ats_issues"),("Recommendations","recommendations")]:
        values=clean_list(analysis.get(key))
        if values:
            story.append(Paragraph(escape(heading),styles["SectionX"]))
            for value in values: story.append(Paragraph("• "+escape(value),styles["BodyX"]))
    story.append(Spacer(1,10))
    story.append(Paragraph("Disclaimer: This report is an AI-generated analysis of the text supplied by the user. It is not a guaranteed ATS score, hiring probability, or professional recruitment decision.",styles["Sub"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ---------- Settings + exports ----------
if page == "Settings":
    render_hero("Settings & Configuration", "Manage your AI connection, model selection information, session data, and privacy expectations.", "PREFERENCES")
    st.write("")
    api_key, model_name=get_config()
    c1,c2=st.columns(2)
    with c1:
        st.markdown("### AI connection")
        if api_key: st.success("API key is configured. The secret value is hidden.")
        else: st.error("GROQ_API_KEY is not configured.")
        st.write("**Selected model**")
        st.code(model_name or "Not configured")
        st.caption("CareerSync uses the exact configured model. It does not silently switch to another model.")
        if st.button("Test Groq connection"):
            try:
                with st.spinner("Checking the configured model..."):
                    response=groq_text("Reply with exactly: CONNECTION_OK","Connection test",max_tokens=20)
                st.success(f"Connection successful. Model response: {response[:100]}")
            except Exception as exc: st.error(str(exc))
    with c2:
        st.markdown("### Privacy & session")
        st.write("• Resume and job text are held in Streamlit session state while this session is active.")
        st.write("• Text is sent to Groq when you request an AI feature.")
        st.write("• Do not upload sensitive information you do not want processed by the AI provider.")
        st.write("• This app does not implement a permanent database.")
        st.write("• Clearing session data removes the app's current in-session workspace.")
    st.markdown("### Clear workspace")
    if st.button("Clear all session data",type="secondary"):
        for key,value in DEFAULT_STATE.items():
            st.session_state[key]=value
        st.rerun()
    st.markdown("### Application information")
    st.write("CareerSync AI · Resume and job matching intelligence · Built with Streamlit, Groq, PyMuPDF, python-docx, Plotly and ReportLab.")
    st.markdown("### PDF analysis report")
    if st.session_state.analysis:
        pdf=build_pdf_report(st.session_state.analysis,st.session_state.job_title,st.session_state.resume_name)
        st.download_button("Download match report (PDF)",pdf,file_name="careersync_match_report.pdf",mime="application/pdf")
    else:
        st.caption("Complete a job match to enable the PDF report download.")


# ---------- Footer ----------
st.markdown("---")
st.markdown('<div style="text-align:center;color:#71819A;font-size:.76rem;padding:8px 0">✦ CareerSync AI · Evidence-based career intelligence · Built for better applications</div>',unsafe_allow_html=True)
