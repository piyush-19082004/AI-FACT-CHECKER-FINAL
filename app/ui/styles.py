"""
Custom CSS styles injected into Streamlit via st.markdown().

All styling is scoped to avoid conflicts with Streamlit's own CSS.
Colors are aligned with the dark theme in .streamlit/config.toml.
"""

# ── Verdict badge colors ───────────────────────────────────────────────────────
VERDICT_COLORS = {
    "Verified":     {"bg": "#14532d", "text": "#4ade80", "border": "#16a34a"},
    "Inaccurate":   {"bg": "#451a03", "text": "#fb923c", "border": "#ea580c"},
    "False":        {"bg": "#450a0a", "text": "#f87171", "border": "#dc2626"},
    "Outdated":     {"bg": "#2e1065", "text": "#c084fc", "border": "#9333ea"},
    "Unverifiable": {"bg": "#1c1917", "text": "#a8a29e", "border": "#57534e"},
}

MAIN_CSS = """
<style>

/* ── Global font & base ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide Streamlit branding ─────────────────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── App header ──────────────────────────────────────────────────────────── */
.fc-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 1.5rem 0 0.5rem 0;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 1.5rem;
}

.fc-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}

.fc-header p {
    color: #64748b;
    font-size: 0.95rem;
    margin: 0;
}

/* ── Upload zone ─────────────────────────────────────────────────────────── */
.fc-upload-hint {
    text-align: center;
    color: #475569;
    font-size: 0.9rem;
    padding: 0.5rem 0;
}

/* ── Verdict badge ───────────────────────────────────────────────────────── */
.verdict-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    border: 1px solid;
    white-space: nowrap;
}

.badge-Verified     { background:#14532d; color:#4ade80; border-color:#16a34a; }
.badge-Inaccurate   { background:#451a03; color:#fb923c; border-color:#ea580c; }
.badge-False        { background:#450a0a; color:#f87171; border-color:#dc2626; }
.badge-Outdated     { background:#2e1065; color:#c084fc; border-color:#9333ea; }
.badge-Unverifiable { background:#1c1917; color:#a8a29e; border-color:#57534e; }

/* ── Metric cards ────────────────────────────────────────────────────────── */
.fc-metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
    margin: 1rem 0;
}

.fc-metric-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    transition: border-color 0.2s;
}

.fc-metric-card:hover { border-color: #334155; }

.fc-metric-value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
}

.fc-metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Confidence bar ──────────────────────────────────────────────────────── */
.conf-bar-wrap {
    background: #1e293b;
    border-radius: 999px;
    height: 6px;
    width: 100%;
    margin: 4px 0;
    overflow: hidden;
}

.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.4s ease;
}

/* ── Claim card ──────────────────────────────────────────────────────────── */
.claim-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 16px 18px;
    margin: 8px 0;
    transition: border-color 0.2s, transform 0.15s;
}

.claim-card:hover {
    border-color: #334155;
    transform: translateY(-1px);
}

.claim-text {
    font-size: 0.95rem;
    color: #e2e8f0;
    line-height: 1.55;
    margin: 6px 0 10px 0;
}

.claim-meta {
    font-size: 0.75rem;
    color: #64748b;
    display: flex;
    gap: 14px;
    align-items: center;
    flex-wrap: wrap;
}

/* ── Evidence source card ────────────────────────────────────────────────── */
.evidence-card {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 12px 14px;
    margin: 6px 0;
}

.evidence-title {
    font-weight: 600;
    font-size: 0.85rem;
    color: #93c5fd;
    margin-bottom: 4px;
}

.evidence-url {
    font-size: 0.72rem;
    color: #4b6a8a;
    word-break: break-all;
    margin-bottom: 6px;
}

.evidence-snippet {
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.5;
}

.evidence-date {
    font-size: 0.7rem;
    color: #475569;
    margin-top: 4px;
}

/* ── Section divider ─────────────────────────────────────────────────────── */
.fc-divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 1.5rem 0;
}

/* ── Status pill ─────────────────────────────────────────────────────────── */
.status-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    background: #1e3a5f;
    color: #60a5fa;
}

/* ── Export button row ───────────────────────────────────────────────────── */
.export-row {
    display: flex;
    gap: 10px;
    margin: 1rem 0;
}

/* ── Scrollable claim list ───────────────────────────────────────────────── */
.claim-scroll {
    max-height: 70vh;
    overflow-y: auto;
    padding-right: 4px;
}

/* ── Sidebar styles ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #070d1a;
    border-right: 1px solid #1e293b;
}

[data-testid="stSidebar"] .stMarkdown h3 {
    color: #93c5fd;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Progress bar ────────────────────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    border-radius: 999px;
}

/* ── Streamlit expander ──────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0f172a;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
}

/* ── Table styling ───────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

</style>
"""


def inject_styles() -> None:
    """Call this once at the top of main.py to apply all custom CSS."""
    import streamlit as st
    st.markdown(MAIN_CSS, unsafe_allow_html=True)


def verdict_badge_html(label: str) -> str:
    """Return an HTML badge string for a verdict label."""
    from app.models.verdict import VERDICT_EMOJI
    from app.models.verdict import VerdictLabel
    try:
        vl    = VerdictLabel(label)
        emoji = VERDICT_EMOJI[vl]
    except (ValueError, KeyError):
        emoji = "❓"
    safe_label = label.replace(" ", "-")
    return (
        f'<span class="verdict-badge badge-{safe_label}">'
        f'{emoji} {label}'
        f'</span>'
    )


def confidence_bar_html(confidence: float, color: str = "#3b82f6") -> str:
    """Return an HTML confidence progress bar."""
    pct = int(confidence * 100)
    return (
        f'<div class="conf-bar-wrap">'
        f'<div class="conf-bar-fill" style="width:{pct}%; background:{color};"></div>'
        f'</div>'
    )
