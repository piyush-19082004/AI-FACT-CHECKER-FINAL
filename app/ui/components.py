"""
Reusable Streamlit UI components for the AI Fact-Checker.

Each function renders one self-contained section of the UI.
Components are pure — they only read data, never mutate session state.
"""

from __future__ import annotations

import json
from typing import Optional

import streamlit as st

from app.models.claim import Claim, FactCheckResult
from app.models.verdict import (
    Verdict, VerdictLabel, EvidenceSource,
    VERDICT_EMOJI, VERDICT_COLOR,
)
from app.ui.styles import verdict_badge_html, confidence_bar_html


# ── Header ────────────────────────────────────────────────────────────────────

def render_header() -> None:
    """Render the app title and subtitle."""
    st.markdown(
        """
        <div class="fc-header">
            <div>
                <h1>🔍 AI Fact-Checker</h1>
                <p>Upload a PDF → extract factual claims → verify against live web sources</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """
    Render the sidebar configuration panel.

    Returns:
        dict with keys: groq_api_key, google_api_key, tavily_api_key,
                        llm_provider, max_claims, max_pages
    """
    import os
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        st.markdown("**🤖 AI Provider**")

        # Read-only status: detect whether keys are set in the environment (for display only)
        env_groq   = os.getenv("GROQ_API_KEY")
        env_google = os.getenv("GOOGLE_API_KEY")
        env_tavily = os.getenv("TAVILY_API_KEY")

        groq_key = ""
        google_key = ""
        tavily_key = ""

        if env_groq:
            llm_provider = "groq"
        elif env_google:
            llm_provider = "google"
        else:
            llm_provider = "groq"

        st.divider()
        st.markdown("**Processing Settings**")
        max_claims = st.slider(
            "Max claims to extract",
            min_value=3, max_value=25, value=10,
            help="More claims = longer processing time and more API calls",
        )
        max_pages = st.slider(
            "Max PDF pages",
            min_value=5, max_value=50, value=20,
            help="Large PDFs are processed up to this page limit",
        )

        st.divider()

        st.markdown("**Status**")
        col1, col2 = st.columns(2)
        with col1:
            if llm_provider == "groq" and env_groq:
                st.success("Groq ✓", icon="🤖")
            elif llm_provider == "google" and env_google:
                st.success("Gemini ✓", icon="🤖")
            else:
                st.error("No LLM key (set via env/st.secrets)", icon="🤖")
        with col2:
            if env_tavily:
                st.success("Tavily ✓", icon="🌐")
            else:
                st.warning("DDG fallback (set TAVILY_API_KEY for Tavily)", icon="🌐")

        st.divider()
        st.markdown(
            "<small style='color:#475569'>Built with Streamlit · LangChain · Groq · Tavily</small>",
            unsafe_allow_html=True,
        )

    return {
        "groq_api_key":   groq_key,
        "google_api_key": google_key if not groq_key else "",
        "tavily_api_key": tavily_key,
        "llm_provider":   llm_provider,
        "max_claims":     max_claims,
        "max_pages":      max_pages,
    }


# ── Upload Section ────────────────────────────────────────────────────────────

def render_upload_section():
    """Render the PDF upload widget and return the uploaded file (or None)."""
    uploaded = st.file_uploader(
        "📄 Upload a PDF document",
        type    = ["pdf"],
        help    = "Text-based PDFs only. Max 50MB. Scanned/image PDFs are not supported.",
        label_visibility = "visible",
    )
    if not uploaded:
        st.markdown(
            '<p class="fc-upload-hint">Supports text-based PDFs · Max 50 MB · '
            'Statistical, scientific, economic, historical claims extracted automatically</p>',
            unsafe_allow_html=True,
        )
    return uploaded


# ── Progress Tracker ──────────────────────────────────────────────────────────

class ProgressTracker:
    """
    Manages Streamlit progress bar + status text during pipeline execution.

    Usage:
        tracker = ProgressTracker()
        tracker.start()
        result  = pipeline.run(..., progress_cb=tracker.update)
        tracker.done()
    """

    def __init__(self):
        self._bar:    Optional[st.delta_generator.DeltaGenerator] = None
        self._status: Optional[st.delta_generator.DeltaGenerator] = None

    def start(self) -> None:
        self._status = st.empty()
        self._bar    = st.progress(0, text="Starting…")

    def update(self, label: str, fraction: float) -> None:
        if self._bar:
            pct = max(0, min(100, int(fraction * 100)))
            self._bar.progress(pct, text=label)
        if self._status:
            self._status.markdown(
                f'<span class="status-pill">{label}</span>',
                unsafe_allow_html=True,
            )

    def done(self) -> None:
        if self._bar:
            self._bar.progress(100, text="✅ Complete!")
        if self._status:
            self._status.empty()

    def clear(self) -> None:
        if self._bar:
            self._bar.empty()
        if self._status:
            self._status.empty()


# ── Summary Metrics ───────────────────────────────────────────────────────────

def render_summary_metrics(result: FactCheckResult) -> None:
    """Render the top-row KPI cards."""
    summary = result.verdict_summary
    total   = result.total_claims
    checked = len(result.checked_claims)

    v_count  = summary.get("Verified",     0)
    i_count  = summary.get("Inaccurate",   0)
    f_count  = summary.get("False",        0)
    o_count  = summary.get("Outdated",     0)
    u_count  = summary.get("Unverifiable", 0)

    cols = st.columns(6)
    metrics = [
        ("Total",        total,   "#60a5fa"),
        ("✅ Verified",   v_count, "#4ade80"),
        ("⚠️ Inaccurate", i_count, "#fb923c"),
        ("❌ False",      f_count, "#f87171"),
        ("🕐 Outdated",   o_count, "#c084fc"),
        ("❓ Unclear",    u_count, "#a8a29e"),
    ]
    for col, (label, value, color) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
                <div class="fc-metric-card">
                    <div class="fc-metric-value" style="color:{color}">{value}</div>
                    <div class="fc-metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Verdict Chart ─────────────────────────────────────────────────────────────

def render_verdict_chart(result: FactCheckResult) -> None:
    """Render a donut chart of verdict distribution using Plotly."""
    import plotly.graph_objects as go

    summary = result.verdict_summary
    if not summary:
        return

    labels = list(summary.keys())
    values = list(summary.values())
    colors = [VERDICT_COLOR.get(VerdictLabel(l), "#6b7280") for l in labels]
    emojis = [VERDICT_EMOJI.get(VerdictLabel(l), "❓") for l in labels]
    display_labels = [f"{e} {l}" for e, l in zip(emojis, labels)]

    fig = go.Figure(go.Pie(
        labels       = display_labels,
        values       = values,
        hole         = 0.55,
        marker       = dict(colors=colors, line=dict(color="#0e1117", width=2)),
        textinfo     = "label+percent",
        textfont     = dict(size=12, color="#e2e8f0"),
        hovertemplate= "<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        showlegend    = False,
        margin        = dict(t=10, b=10, l=10, r=10),
        height        = 280,
        annotations   = [dict(
            text      = f"<b>{result.total_claims}</b><br><span style='font-size:11px'>claims</span>",
            x=0.5, y=0.5,
            font      = dict(size=18, color="#e2e8f0"),
            showarrow = False,
        )],
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Category Distribution Chart ───────────────────────────────────────────────

def render_category_chart(result: FactCheckResult) -> None:
    """Render a horizontal bar chart of claims by category."""
    import plotly.graph_objects as go
    from collections import Counter

    cats = Counter(c.category.value for c in result.claims)
    if not cats:
        return

    sorted_cats   = sorted(cats.items(), key=lambda x: x[1])
    cat_labels    = [c[0] for c in sorted_cats]
    cat_values    = [c[1] for c in sorted_cats]

    fig = go.Figure(go.Bar(
        x           = cat_values,
        y           = cat_labels,
        orientation = "h",
        marker      = dict(
            color     = "#3b82f6",
            line      = dict(color="#1d4ed8", width=1),
        ),
        text        = cat_values,
        textposition= "auto",
        textfont    = dict(color="#e2e8f0", size=11),
        hovertemplate = "<b>%{y}</b>: %{x} claims<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        margin        = dict(t=5, b=5, l=5, r=30),
        height        = max(160, len(cat_labels) * 38),
        xaxis         = dict(
            showgrid      = True,
            gridcolor     = "#1e293b",
            color         = "#64748b",
            tickfont      = dict(size=10),
        ),
        yaxis         = dict(color="#94a3b8", tickfont=dict(size=11)),
        bargap        = 0.35,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Individual Claim Card ─────────────────────────────────────────────────────

def render_claim_card(claim: Claim, index: int) -> None:
    """
    Render one claim with its verdict badge, confidence bar,
    explanation, and collapsible evidence sources.
    """
    verdict = claim.verdict
    label   = verdict.label.value if verdict else "Pending"
    color   = verdict.color       if verdict else "#6b7280"
    badge   = verdict_badge_html(label)
    conf    = verdict.confidence  if verdict else 0.0

    # ── Outer card HTML ───────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="claim-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px;">
                <div style="flex:1">
                    <div class="claim-meta">
                        <span style="color:#60a5fa; font-weight:600">#{claim.id}</span>
                        <span>📄 Page {claim.page_number}</span>
                        <span>🏷️ {claim.category.value}</span>
                    </div>
                    <div class="claim-text">{claim.text}</div>
                </div>
                <div style="text-align:right; min-width:110px;">
                    {badge}
                    <div style="font-size:0.7rem; color:#64748b; margin-top:4px;">{int(conf*100)}% confident</div>
                    {confidence_bar_html(conf, color)}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Expandable detail ─────────────────────────────────────────────────────
    if verdict:
        with st.expander(f"📋 Evidence & Explanation — Claim #{claim.id}", expanded=False):
            # Explanation
            st.markdown(f"**🤖 Reasoning**")
            st.markdown(
                f'<div style="background:#0a1628; border-left:3px solid {color}; '
                f'padding:12px 14px; border-radius:0 6px 6px 0; '
                f'font-size:0.88rem; color:#cbd5e1; line-height:1.6;">'
                f'{verdict.explanation}'
                f'</div>',
                unsafe_allow_html=True,
            )

            if claim.search_query:
                st.markdown(
                    f'<div style="margin-top:8px; font-size:0.75rem; color:#475569;">'
                    f'🔎 Search query: <code style="color:#93c5fd">{claim.search_query}</code>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Evidence sources
            if verdict.sources:
                st.markdown(f"**🌐 Evidence Sources ({len(verdict.sources)})**")
                for i, src in enumerate(verdict.sources, 1):
                    render_evidence_source(src, index=i)
            else:
                st.info("No web sources were retrieved for this claim.", icon="🔍")

            # Context from PDF
            if claim.context:
                st.markdown("**📄 Original Context**")
                st.markdown(
                    f'<div style="background:#0f172a; border:1px solid #1e293b; '
                    f'border-radius:6px; padding:10px 12px; font-size:0.82rem; '
                    f'color:#94a3b8; font-style:italic; line-height:1.6;">'
                    f'"{claim.context}"'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def render_evidence_source(src: EvidenceSource, index: int) -> None:
    """Render a single evidence source card."""
    date_html = (
        f'<div class="evidence-date">📅 {src.date}</div>'
        if src.date else ""
    )
    domain = src.url.split("/")[2] if "/" in src.url else src.url

    st.markdown(
        f"""
        <div class="evidence-card">
            <div class="evidence-title">{index}. {src.title}</div>
            <div class="evidence-url">🔗 <a href="{src.url}" target="_blank"
                style="color:#4b6a8a; text-decoration:none;">{domain}</a></div>
            <div class="evidence-snippet">{src.snippet}</div>
            {date_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Claims Table (Compact View) ───────────────────────────────────────────────

def render_claims_table(result: FactCheckResult) -> None:
    """Render a compact, filterable Pandas DataFrame of all claims."""
    import pandas as pd

    df = result.to_dataframe()
    if df.empty:
        st.warning("No claims to display.", icon="🤷")
        return

    # ── Filters ───────────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])

    with col_f1:
        all_verdicts = ["All"] + sorted(df["verdict"].unique().tolist())
        verdict_filter = st.selectbox("Filter by verdict", all_verdicts, index=0)

    with col_f2:
        all_cats = ["All"] + sorted(df["category"].unique().tolist())
        cat_filter = st.selectbox("Filter by category", all_cats, index=0)

    with col_f3:
        sort_by = st.selectbox("Sort by", ["id", "confidence", "category", "page"])

    # Apply filters
    filtered = df.copy()
    if verdict_filter != "All":
        filtered = filtered[filtered["verdict"] == verdict_filter]
    if cat_filter != "All":
        filtered = filtered[filtered["category"] == cat_filter]

    # Sort
    if sort_by == "confidence":
        # Parse "XX%" back to float for sorting
        filtered["_conf_sort"] = filtered["confidence"].str.replace("%", "").replace("—", "0").astype(float)
        filtered = filtered.sort_values("_conf_sort", ascending=False).drop(columns=["_conf_sort"])
    elif sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by)

    st.caption(f"Showing {len(filtered)} of {len(df)} claims")

    st.dataframe(
        filtered[["id", "claim", "category", "page", "verdict", "confidence"]],
        use_container_width = True,
        hide_index          = True,
        column_config       = {
            "id":         st.column_config.NumberColumn("#",          width="small"),
            "claim":      st.column_config.TextColumn("Claim",        width="large"),
            "category":   st.column_config.TextColumn("Category",     width="medium"),
            "page":       st.column_config.NumberColumn("Page",       width="small"),
            "verdict":    st.column_config.TextColumn("Verdict",      width="medium"),
            "confidence": st.column_config.TextColumn("Confidence",   width="small"),
        },
    )


# ── Claims Detailed View ──────────────────────────────────────────────────────

def render_claims_detailed(result: FactCheckResult) -> None:
    """Render all claims as expandable cards with full evidence."""
    if not result.claims:
        st.warning("No claims found.", icon="🤷")
        return

    # Verdict filter chips
    st.markdown("**Filter by verdict:**")
    all_labels = [v.value for v in VerdictLabel]
    selected_labels = []

    cols = st.columns(len(all_labels))
    for i, (col, label) in enumerate(zip(cols, all_labels)):
        with col:
            count = sum(
                1 for c in result.claims
                if c.verdict and c.verdict.label.value == label
            )
            emoji = VERDICT_EMOJI.get(VerdictLabel(label), "❓")
            if st.checkbox(f"{emoji} {label} ({count})", value=True, key=f"filter_{label}"):
                selected_labels.append(label)

    st.markdown('<hr class="fc-divider">', unsafe_allow_html=True)

    # Render filtered claims
    shown = 0
    for claim in result.claims:
        verdict_label = claim.verdict.label.value if claim.verdict else "Unverifiable"
        if verdict_label in selected_labels or not claim.verdict:
            render_claim_card(claim, index=claim.id)
            shown += 1

    if shown == 0:
        st.info("No claims match the selected filters.", icon="🔍")


# ── Export Panel ──────────────────────────────────────────────────────────────

def render_export_panel(result: FactCheckResult) -> None:
    """Render CSV and JSON export buttons."""
    import pandas as pd

    df = result.to_dataframe()

    # Build rich JSON export
    rich_export = {
        "filename":     result.filename,
        "total_pages":  result.total_pages,
        "total_claims": result.total_claims,
        "verdict_summary": result.verdict_summary,
        "claims": [
            {
                "id":       c.id,
                "text":     c.text,
                "category": c.category.value,
                "page":     c.page_number,
                "context":  c.context,
                "search_query": c.search_query,
                "verdict": {
                    "label":       c.verdict.label.value,
                    "confidence":  c.verdict.confidence,
                    "explanation": c.verdict.explanation,
                    "sources": [
                        {
                            "url":     s.url,
                            "title":   s.title,
                            "snippet": s.snippet,
                            "date":    s.date,
                        }
                        for s in c.verdict.sources
                    ],
                } if c.verdict else None,
            }
            for c in result.claims
        ],
    }

    json_str = json.dumps(rich_export, indent=2, ensure_ascii=False)
    csv_str  = df.to_csv(index=False)

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.download_button(
            label     = "⬇️ Download CSV",
            data      = csv_str,
            file_name = f"{result.filename}_fact_check.csv",
            mime      = "text/csv",
            use_container_width = True,
        )
    with col2:
        st.download_button(
            label     = "⬇️ Download JSON",
            data      = json_str,
            file_name = f"{result.filename}_fact_check.json",
            mime      = "application/json",
            use_container_width = True,
        )
    with col3:
        st.caption(
            f"CSV: flat table · JSON: full detail with evidence sources and explanations"
        )


# ── PDF Info Banner ───────────────────────────────────────────────────────────

def render_pdf_info(extraction_result, filename: str) -> None:
    """Render a compact summary of the uploaded PDF."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Filename",       filename[:28] + "…" if len(filename) > 28 else filename)
    col2.metric("📑 Pages Processed", extraction_result.text_page_count)
    col3.metric("📊 Total Pages",     extraction_result.total_pages)
    col4.metric("🔤 Characters",      f"{len(extraction_result.full_text):,}")

    for warning in extraction_result.warnings:
        st.warning(warning, icon="⚠️")


# ── Empty State ───────────────────────────────────────────────────────────────

def render_empty_state() -> None:
    """Render the landing page before any file is uploaded."""
    st.markdown(
        """
        <div style="text-align:center; padding: 3rem 1rem;">
            <div style="font-size:4rem; margin-bottom:1rem;">📄</div>
            <h3 style="color:#475569; font-weight:500; margin-bottom:0.5rem;">
                Upload a PDF to begin
            </h3>
            <p style="color:#334155; max-width:420px; margin:0 auto; line-height:1.6;">
                The app will automatically extract factual claims, search the web
                for evidence, and return a verdict for each claim.
            </p>
        </div>

        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-top:2rem;">
            <div class="fc-metric-card">
                <div style="font-size:1.8rem; margin-bottom:8px;">🧠</div>
                <div style="font-weight:600; color:#e2e8f0; margin-bottom:4px;">Claim Extraction</div>
                <div style="font-size:0.8rem; color:#64748b;">
                    Gemini 1.5 Flash identifies all verifiable claims
                </div>
            </div>
            <div class="fc-metric-card">
                <div style="font-size:1.8rem; margin-bottom:8px;">🌐</div>
                <div style="font-weight:600; color:#e2e8f0; margin-bottom:4px;">Live Web Search</div>
                <div style="font-size:0.8rem; color:#64748b;">
                    Tavily retrieves real-time evidence per claim
                </div>
            </div>
            <div class="fc-metric-card">
                <div style="font-size:1.8rem; margin-bottom:8px;">⚖️</div>
                <div style="font-weight:600; color:#e2e8f0; margin-bottom:4px;">Verdict Engine</div>
                <div style="font-size:0.8rem; color:#64748b;">
                    Gemini 1.5 Pro reasons and classifies each claim
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
