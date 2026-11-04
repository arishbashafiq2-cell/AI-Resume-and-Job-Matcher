import io
import json
import re
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq


# ------------------------------------------------------------
# Application configuration
# ------------------------------------------------------------
APP_TITLE = "TalentLens AI"
GROQ_MODEL = "openai/gpt-oss-120b"


# ------------------------------------------------------------
# 1. Enterprise styling
# ------------------------------------------------------------
def init_css() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #0E1117;
            --panel: #151B26;
            --panel-2: #192231;
            --line: rgba(148, 163, 184, .16);
            --muted: #91A0B5;
            --text: #F1F5F9;
            --blue: #0066FF;
            --green: #10B981;
            --red: #FB7185;
            --amber: #FBBF24;
        }

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at 82% 0%, rgba(0,102,255,.10), transparent 27%),
                radial-gradient(circle at 4% 30%, rgba(16,185,129,.05), transparent 24%),
                var(--bg);
            color: var(--text);
        }
        [data-testid="stHeader"] { background: rgba(14,17,23,.82); }
        [data-testid="stSidebar"] {
            background: #111722;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] > div { padding-top: 1.5rem; }
        .block-container {
            max-width: 1450px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }
        h1, h2, h3, h4 {
            font-family: 'Manrope', sans-serif !important;
            letter-spacing: -.035em;
            color: #F8FAFC;
        }
        p, li, label { color: #CBD5E1; }
        .hero {
            position: relative;
            overflow: hidden;
            padding: 30px 32px;
            border: 1px solid rgba(96,165,250,.20);
            border-radius: 22px;
            background: linear-gradient(120deg, rgba(21,35,57,.98), rgba(17,24,39,.94));
            box-shadow: 0 18px 55px rgba(0,0,0,.25);
            margin-bottom: 26px;
        }
        .hero:after {
            content: "";
            position: absolute;
            width: 260px; height: 260px;
            right: -70px; top: -130px;
            border-radius: 50%;
            background: rgba(0,102,255,.16);
            filter: blur(3px);
        }
        .eyebrow {
            color: #60A5FA;
            font-size: 11px;
            letter-spacing: .18em;
            text-transform: uppercase;
            font-weight: 700;
        }
        .hero-title {
            font-family: 'Manrope', sans-serif;
            font-size: clamp(30px, 4vw, 43px);
            font-weight: 800;
            line-height: 1.12;
            margin: 10px 0 8px;
            color: #F8FAFC;
        }
        .hero-copy { color: #A9B8CC; font-size: 15px; max-width: 720px; }
        .section-kicker {
            color: #60A5FA;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: .15em;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .section-heading {
            color: #F8FAFC;
            font-family: 'Manrope', sans-serif;
            font-weight: 750;
            font-size: 23px;
            margin: 0 0 14px;
        }
        .metric-card {
            background: linear-gradient(145deg, #192231, #141B26);
            border: 1px solid var(--line);
            border-radius: 17px;
            padding: 19px 20px;
            min-height: 125px;
            box-shadow: 0 12px 32px rgba(0,0,0,.16);
        }
        .metric-label {
            color: #91A0B5;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: .04em;
        }
        .metric-value {
            color: #F8FAFC;
            font-family: 'Manrope', sans-serif;
            font-size: 31px;
            font-weight: 800;
            margin-top: 8px;
            line-height: 1.1;
        }
        .metric-note { color: #8292A8; font-size: 11px; margin-top: 8px; }
        .panel {
            background: rgba(21,27,38,.88);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 21px;
            box-shadow: 0 12px 35px rgba(0,0,0,.13);
        }
        .tag {
            display: inline-block;
            padding: 6px 10px;
            margin: 4px 5px 4px 0;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            background: rgba(16,185,129,.11);
            color: #6EE7B7;
            border: 1px solid rgba(16,185,129,.22);
        }
        .tag-missing {
            background: rgba(251,113,133,.09);
            color: #FDA4AF;
            border-color: rgba(251,113,133,.22);
        }
        .improvement-card {
            background: #151D29;
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 16px 18px;
            margin: 10px 0;
        }
        .improvement-label {
            color: #7F91AA;
            text-transform: uppercase;
            letter-spacing: .1em;
            font-size: 10px;
            font-weight: 700;
            margin-bottom: 5px;
        }
        .improvement-new { color: #A7F3D0; font-weight: 600; }
        .improvement-old { color: #CBD5E1; }
        .improvement-why { color: #91A0B5; font-size: 13px; margin-top: 9px; }
        .stTextArea textarea, .stTextInput input {
            background: #111925 !important;
            color: #F1F5F9 !important;
            border: 1px solid rgba(148,163,184,.22) !important;
            border-radius: 12px !important;
        }
        .stSelectbox div[data-baseweb="select"] > div {
            background: #111925;
            border-color: rgba(148,163,184,.22);
            border-radius: 11px;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 11px;
            border: 1px solid rgba(96,165,250,.35);
            background: linear-gradient(110deg, #0066FF, #155EEF);
            color: white;
            font-weight: 700;
            padding: .65rem 1.1rem;
            transition: all .2s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: #60A5FA;
            box-shadow: 0 7px 22px rgba(0,102,255,.24);
            transform: translateY(-1px);
            color: white;
        }
        [data-testid="stDataFrame"], [data-testid="stDataEditor"] {
            border: 1px solid var(--line);
            border-radius: 12px;
            overflow: hidden;
        }
        div[data-testid="stAlert"] { border-radius: 12px; }
        hr { border-color: var(--line); }
        #MainMenu, footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _clean_json_response(raw_text: str) -> dict[str, Any]:
    """Parse JSON returned by the model, including fenced JSON responses."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("The AI response did not contain a valid JSON object.")

    parsed = json.loads(cleaned[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("The AI response must be a JSON object.")

    required = {
        "match_score",
        "matching_skills",
        "missing_requirements",
        "wording_improvements",
    }
    missing_keys = required.difference(parsed.keys())
    if missing_keys:
        raise ValueError(
            "The AI response is missing required fields: "
            + ", ".join(sorted(missing_keys))
        )

    score = parsed["match_score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("The match score must be a number.")
    parsed["match_score"] = max(0, min(100, int(round(score))))

    for key in ("matching_skills", "missing_requirements", "wording_improvements"):
        if not isinstance(parsed[key], list):
            raise ValueError(f"The '{key}' field must be an array.")

    parsed["matching_skills"] = [
        str(item).strip() for item in parsed["matching_skills"] if str(item).strip()
    ]
    parsed["missing_requirements"] = [
        str(item).strip()
        for item in parsed["missing_requirements"]
        if str(item).strip()
    ]

    normalized_improvements = []
    for item in parsed["wording_improvements"]:
        if not isinstance(item, dict):
            continue
        normalized_improvements.append({
            "current_phrasing": str(item.get("current_phrasing", "")).strip(),
            "suggested_phrasing": str(item.get("suggested_phrasing", "")).strip(),
            "why_it_helps": str(item.get("why_it_helps", "")).strip(),
        })
    parsed["wording_improvements"] = normalized_improvements

    return parsed


# ------------------------------------------------------------
# 2. Groq analysis engine
# ------------------------------------------------------------
def analyze_resume_match(
    resume_text: str,
    job_desc: str,
    api_key: str,
    target_role: str,
    role_level: str,
) -> dict[str, Any]:
    """Analyze a resume against a job description using the configured Groq model."""
    system_prompt = """
You are TalentLens AI, an expert ATS analyst, executive recruiter, and resume
optimization specialist.

Compare the candidate's resume against the supplied job description. Be precise,
evidence-based, fair, and specific. Do not invent experience, qualifications,
certifications, achievements, metrics, employers, or skills. Do not reward a
candidate for a skill that is not supported by the resume. Distinguish explicit
evidence from reasonable inference. A match score is an approximate relevance
assessment, not a prediction of hiring success.

Return ONLY one valid JSON object, with no markdown fences or surrounding prose,
using exactly this schema:
{
  "match_score": 0,
  "matching_skills": ["skill supported by resume and relevant to job"],
  "missing_requirements": ["important job requirement not evidenced in resume"],
  "wording_improvements": [
    {
      "current_phrasing": "short exact or faithful excerpt from the resume",
      "suggested_phrasing": "truthful, clearer, role-aligned rewrite",
      "why_it_helps": "brief explanation of relevance, clarity, or ATS terminology"
    }
  ]
}

Rules:
- match_score must be an integer from 0 to 100.
- matching_skills must include only relevant skills supported by resume evidence.
- missing_requirements must identify relevant gaps or absent keywords, not assume
  the candidate lacks a skill merely because the resume does not mention it.
- wording_improvements should contain up to 8 useful edits. If there is no
  meaningful source wording to improve, return an empty array.
- Never fabricate numerical results. Use [add verified metric] only when a
  metric would genuinely improve the bullet and explain that the user must
  supply a real value.
- Keep suggestions professional, concise, ATS-readable, and faithful to facts.
- Consider the target role and seniority context when supplied.
- Do not include personal demographic or protected-class assumptions.
"""

    user_prompt = f"""
TARGET ROLE / INDUSTRY: {target_role}
ROLE LEVEL: {role_level}

CANDIDATE RESUME:
<resume>
{resume_text}
</resume>

JOB DESCRIPTION:
<job_description>
{job_desc}
</job_description>

Assess alignment and return the required JSON only.
"""

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_completion_tokens=3000,
            response_format={"type": "json_object"},
        )

        content = completion.choices[0].message.content
        if not content:
            raise ValueError("The AI returned an empty response.")

        return _clean_json_response(content)

    except Exception as exc:
        message = str(exc)
        lowered = message.lower()

        if "401" in lowered or "authentication" in lowered or "invalid api key" in lowered:
            raise RuntimeError(
                "Groq authentication failed. Check your GROQ_API_KEY secret or sidebar key."
            ) from exc
        if "model" in lowered and (
            "not found" in lowered or "decommission" in lowered or "does not exist" in lowered
        ):
            raise RuntimeError(
                f"The configured Groq model '{GROQ_MODEL}' is unavailable for this API key. "
                "Check Groq's current model catalogue and update GROQ_MODEL in app.py."
            ) from exc
        if "429" in lowered or "rate limit" in lowered:
            raise RuntimeError(
                "Groq rate limit reached. Please wait a little and try again."
            ) from exc
        if isinstance(exc, (ValueError, json.JSONDecodeError)):
            raise RuntimeError(f"Could not validate the AI response: {exc}") from exc

        raise RuntimeError(f"Resume analysis failed: {message}") from exc


# ------------------------------------------------------------
# UI rendering helpers
# ------------------------------------------------------------
def render_metric(label: str, value: str, note: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{accent}">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tags(items: list[str], missing: bool = False) -> None:
    if not items:
        st.caption("No items identified.")
        return
    tag_class = "tag tag-missing" if missing else "tag"
    html = "".join(
        f'<span class="{tag_class}">{_html_escape(item)}</span>'
        for item in items
    )
    st.markdown(html, unsafe_allow_html=True)


def _html_escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def render_gauge(score: int) -> None:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%", "font": {"size": 43, "color": "#F8FAFC"}},
            title={"text": "Overall resume alignment", "font": {"size": 15, "color": "#A9B8CC"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#64748B",
                    "tickfont": {"color": "#91A0B5"},
                },
                "bar": {"color": "#10B981", "thickness": 0.24},
                "bgcolor": "#151B26",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#263242"},
                    {"range": [40, 70], "color": "#23364A"},
                    {"range": [70, 100], "color": "#1D3A3A"},
                ],
                "threshold": {
                    "line": {"color": "#60A5FA", "width": 4},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )
    figure.update_layout(
        height=280,
        margin=dict(l=25, r=25, t=55, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans"},
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


# ------------------------------------------------------------
# 3. Main application
# ------------------------------------------------------------
def main() -> None:
    init_css()

    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "analysis_signature" not in st.session_state:
        st.session_state.analysis_signature = None
    if "editable_improvements" not in st.session_state:
        st.session_state.editable_improvements = None

    # Sidebar
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:11px;margin-bottom:20px">
                <div style="width:39px;height:39px;border-radius:12px;
                    background:linear-gradient(135deg,#0066FF,#10B981);
                    display:flex;align-items:center;justify-content:center;
                    color:white;font-size:21px;font-weight:800">✦</div>
                <div>
                    <div style="font-family:Manrope;font-size:18px;font-weight:800;color:#F8FAFC">
                        TalentLens AI
                    </div>
                    <div style="font-size:10px;color:#8292A8;letter-spacing:.12em">
                        CAREER INTELLIGENCE
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Workspace")
        st.caption("Configure your analysis context.")
        target_role = st.selectbox(
            "Target industry / role",
            [
                "General / Not specified",
                "Software Engineering",
                "Web Development",
                "Data Science & Analytics",
                "Cybersecurity",
                "Product Management",
                "UI/UX & Graphic Design",
                "Marketing & Communications",
                "Finance & Accounting",
                "Human Resources",
                "Operations & Project Management",
                "Other",
            ],
        )
        role_level = st.selectbox(
            "Target role level",
            ["Not specified", "Internship", "Entry level", "Mid-level", "Senior", "Lead / Manager", "Executive"],
        )
        st.divider()
        st.markdown("### AI configuration")
        try:
            secret_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            secret_key = ""

        if secret_key:
            api_key = secret_key
            st.success("Groq API key loaded from Streamlit Secrets.")
        else:
            api_key = st.text_input(
                "Groq API key",
                type="password",
                help="Paste your Groq API key. It is used for this session only.",
                placeholder="gsk_...",
            )
            st.caption("For deployment, add GROQ_API_KEY in Streamlit Secrets.")
        st.markdown(
            f"""
            <div style="padding:12px;border:1px solid rgba(148,163,184,.15);
                border-radius:12px;background:#151D29;margin-top:12px">
                <div style="font-size:10px;color:#8292A8;letter-spacing:.1em">ACTIVE MODEL</div>
                <div style="font-size:12px;color:#93C5FD;font-weight:700;margin-top:5px">
                    {GROQ_MODEL}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.caption("TalentLens AI · Resume intelligence workspace")

    # Header
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">✦ AI-powered talent intelligence</div>
            <div class="hero-title">Make your next move<br>with confidence.</div>
            <div class="hero-copy">
                Compare your resume with a target role, uncover skill gaps,
                and transform your experience into clear, compelling language.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input workspace
    st.markdown('<div class="section-kicker">01 / INPUT WORKSPACE</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Build your match analysis</div>', unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Your resume**")
        uploaded_file = st.file_uploader(
            "Upload a plain-text resume",
            type=["txt", "md"],
            help="Supported formats: TXT and Markdown. You can also paste your resume below.",
        )
        uploaded_text = ""
        if uploaded_file is not None:
            try:
                uploaded_text = uploaded_file.getvalue().decode("utf-8-sig")
                st.success(f"Loaded: {uploaded_file.name}")
            except UnicodeDecodeError:
                st.error("This file could not be read as UTF-8 text. Please save it as UTF-8 or paste the resume.")
        resume_text = st.text_area(
            "Paste or edit resume text",
            value=uploaded_text,
            height=270,
            placeholder="Paste your professional summary, experience, skills, education, projects, and certifications...",
            label_visibility="collapsed",
        )
        st.caption(f"{len(resume_text.split())} words · Keep personal contact details out if not needed.")

    with right:
        st.markdown("**Target job description**")
        job_desc = st.text_area(
            "Job description",
            height=350,
            placeholder="Paste the complete job description, including responsibilities, required skills, qualifications, and preferred experience...",
            label_visibility="collapsed",
        )
        st.caption(f"{len(job_desc.split())} words · Include the full requirements for a more useful comparison.")

    button_col, reset_col, _ = st.columns([1.25, 1, 3])
    with button_col:
        analyze_clicked = st.button("✦  Analyze my match", type="primary", use_container_width=True)
    with reset_col:
        reset_clicked = st.button("Reset analysis", use_container_width=True)

    if reset_clicked:
        st.session_state.analysis_result = None
        st.session_state.analysis_signature = None
        st.session_state.editable_improvements = None
        st.rerun()

    if analyze_clicked:
        if not resume_text.strip() or not job_desc.strip():
            st.warning("Please provide both your resume and the job description before starting the analysis.")
        elif not api_key.strip():
            st.warning("Add your Groq API key in the sidebar or configure GROQ_API_KEY in Streamlit Secrets.")
        elif len(resume_text.strip()) < 50:
            st.warning("Your resume text is very short. Add more detail so the analysis can be meaningful.")
        elif len(job_desc.strip()) < 50:
            st.warning("The job description is very short. Please provide the role requirements and responsibilities.")
        else:
            signature = (
                resume_text.strip(),
                job_desc.strip(),
                target_role,
                role_level,
            )
            with st.spinner("Reading your experience and mapping it to the role..."):
                try:
                    result = analyze_resume_match(
                        resume_text=resume_text.strip(),
                        job_desc=job_desc.strip(),
                        api_key=api_key.strip(),
                        target_role=target_role,
                        role_level=role_level,
                    )
                    st.session_state.analysis_result = result
                    st.session_state.analysis_signature = signature
                    st.session_state.editable_improvements = pd.DataFrame(
                        result["wording_improvements"],
                        columns=["current_phrasing", "suggested_phrasing", "why_it_helps"],
                    )
                    st.success("Your resume analysis is ready.")
                except Exception as exc:
                    st.error(str(exc))

    result = st.session_state.analysis_result

    if result is None:
        st.markdown("---")
        st.markdown(
            """
            <div class="panel" style="text-align:center;padding:34px 20px">
                <div style="font-size:34px;margin-bottom:10px">◎</div>
                <div style="font-family:Manrope;font-size:19px;font-weight:750;color:#F8FAFC">
                    Your insights will appear here
                </div>
                <div style="color:#91A0B5;font-size:13px;margin-top:7px">
                    Add your resume and a job description, then run the analysis to unlock your report.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Report
    st.markdown("---")
    st.markdown('<div class="section-kicker">02 / MATCH OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Your role alignment report</div>', unsafe_allow_html=True)

    score = result["match_score"]
    matching = result["matching_skills"]
    missing = result["missing_requirements"]
    improvements = result["wording_improvements"]

    k1, k2, k3 = st.columns(3, gap="medium")
    with k1:
        render_metric("ATS MATCH SCORE", f"{score}%", "Estimated alignment with this job description", "#10B981")
    with k2:
        render_metric("MATCHED CORE SKILLS", str(len(matching)), "Relevant skills evidenced in your resume", "#60A5FA")
    with k3:
        render_metric("SKILL GAPS", str(len(missing)), "Requirements not clearly evidenced", "#FB7185")

    chart_col, insight_col = st.columns([1, 1.25], gap="large")
    with chart_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        render_gauge(score)
        st.markdown("</div>", unsafe_allow_html=True)
    with insight_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Match interpretation")
        if score >= 80:
            st.success("Strong alignment: your resume reflects many of the role's stated requirements.")
        elif score >= 60:
            st.info("Moderate alignment: several relevant signals are present, with opportunities to improve.")
        elif score >= 40:
            st.warning("Developing alignment: important requirements need clearer evidence or further development.")
        else:
            st.error("Low alignment: the current resume shows limited evidence for this particular role.")
        st.markdown(
            "The score is an AI-generated estimate based on the supplied text. "
            "It is not an official ATS result or a hiring decision."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Skill analysis
    st.markdown("")
    st.markdown('<div class="section-kicker">03 / SKILL INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Where you align — and where to focus</div>', unsafe_allow_html=True)
    match_col, gap_col = st.columns(2, gap="large")
    with match_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### <span style='color:#6EE7B7'>✓</span> Matching skills", unsafe_allow_html=True)
        st.caption("Relevant capabilities supported by your resume.")
        render_tags(matching)
        st.markdown("</div>", unsafe_allow_html=True)
    with gap_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### <span style='color:#FDA4AF'>＋</span> Missing / unverified requirements", unsafe_allow_html=True)
        st.caption("Important job requirements not clearly demonstrated in the resume.")
        render_tags(missing, missing=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if matching or missing:
        chart_data = pd.DataFrame(
            {
                "Category": ["Matched skills", "Unverified requirements"],
                "Count": [len(matching), len(missing)],
            }
        )
        fig = px.bar(
            chart_data,
            x="Count",
            y="Category",
            orientation="h",
            text="Count",
            color="Category",
            color_discrete_map={
                "Matched skills": "#10B981",
                "Unverified requirements": "#FB7185",
            },
        )
        fig.update_layout(
            height=190,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1",
            showlegend=False,
            margin=dict(l=10, r=20, t=10, b=10),
            xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,.12)", zeroline=False),
            yaxis=dict(showgrid=False, title=""),
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Wording optimization
    st.markdown("---")
    st.markdown('<div class="section-kicker">04 / RESUME OPTIMIZATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Upgrade your wording</div>', unsafe_allow_html=True)
    st.caption(
        "Review and edit the AI suggestions before exporting. Only use wording that accurately reflects your real experience."
    )

    if improvements:
        if st.session_state.editable_improvements is None:
            st.session_state.editable_improvements = pd.DataFrame(
                improvements,
                columns=["current_phrasing", "suggested_phrasing", "why_it_helps"],
            )

        edited_df = st.data_editor(
            st.session_state.editable_improvements,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key="improvements_editor",
            column_config={
                "current_phrasing": st.column_config.TextColumn(
                    "Current phrasing", width="medium"
                ),
                "suggested_phrasing": st.column_config.TextColumn(
                    "Suggested phrasing", width="large"
                ),
                "why_it_helps": st.column_config.TextColumn(
                    "Why it helps", width="large"
                ),
            },
        )
        st.session_state.editable_improvements = edited_df

        csv_bytes = edited_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "↓  Download wording improvements (CSV)",
            data=csv_bytes,
            file_name="talentlens_resume_improvements.csv",
            mime="text/csv",
        )

        with st.expander("View polished wording cards"):
            for index, row in edited_df.iterrows():
                st.markdown(
                    f"""
                    <div class="improvement-card">
                        <div class="improvement-label">Original wording · {index + 1}</div>
                        <div class="improvement-old">{_html_escape(row.get('current_phrasing', ''))}</div>
                        <div class="improvement-label" style="margin-top:13px">Suggested upgrade</div>
                        <div class="improvement-new">{_html_escape(row.get('suggested_phrasing', ''))}</div>
                        <div class="improvement-why">{_html_escape(row.get('why_it_helps', ''))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("The AI did not identify specific wording changes. Your resume may already communicate its relevant experience clearly, or there may not be enough source wording to improve.")

    st.markdown("---")
    st.caption(
        "Privacy note: resume text is sent to Groq to generate the analysis. "
        "Avoid including sensitive personal information that is not needed for the comparison."
    )


if __name__ == "__main__":
    main()
