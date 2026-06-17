from __future__ import annotations

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return None

load_dotenv()

st.set_page_config(
    page_title            = "AI Fact-Checker",
    page_icon             = "🔍",
    layout                = "wide",
    initial_sidebar_state = "expanded",
    menu_items            = {
        "Get help":     "https://github.com/your-username/fact-checker",
        "Report a Bug": "https://github.com/your-username/fact-checker/issues",
        "About":        "AI Fact-Checker — Streamlit · LangChain · Gemini · Tavily",
    },
)

from app.ui.styles import inject_styles
inject_styles()

from app.ui.components import (
    render_header, render_sidebar, render_upload_section,
    render_pdf_info, render_summary_metrics, render_verdict_chart,
    render_category_chart, render_claims_table, render_claims_detailed,
    render_export_panel, render_empty_state, ProgressTracker,
)
from app.core.pdf_extractor import (
    PDFExtractionError, ScannedPDFError, EmptyPDFError,
)
from app.core.pipeline import FactCheckPipeline, PipelineConfig
from app.models.claim import FactCheckResult
from app.utils.validators import (
    validate_pdf_file, FileTooLargeError,
    InvalidFileTypeError, EmptyFileError, APIKeyValidator,
)
from app.config import get_settings




def _init_session() -> None:
    defaults = {
        "result":        None,
        "extraction":    None,
        "uploaded_name": None,
        "is_processing": False,
        "process_error": None,
        "run_count":     0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()
_cfg = get_settings()




render_header()
sidebar_cfg = render_sidebar()


def _resolve(sidebar_val: str, env_key: str) -> str:
    if sidebar_val and sidebar_val.strip():
        return sidebar_val.strip()
    if env := os.getenv(env_key, ""):
        return env
    try:
        return st.secrets.get(env_key, "")
    except Exception:
        return ""

groq_key   = _resolve(sidebar_cfg["groq_api_key"],   "GROQ_API_KEY")
google_key = _resolve(sidebar_cfg["google_api_key"], "GOOGLE_API_KEY")
tavily_key = _resolve(sidebar_cfg["tavily_api_key"], "TAVILY_API_KEY")
llm_provider = sidebar_cfg["llm_provider"]

_kv = APIKeyValidator()
if tavily_key:
    check = _kv.check_tavily_key(tavily_key)
    if not check.is_valid:
        st.warning(f"⚠️ **Tavily key issue:** {check.warning} (will use DuckDuckGo fallback)", icon="🌐")
        tavily_key = ""




uploaded_file = render_upload_section()

if not uploaded_file:
    render_empty_state()
    st.stop()

pdf_bytes = uploaded_file.read()

if st.session_state.uploaded_name != uploaded_file.name:
    st.session_state.result        = None
    st.session_state.extraction    = None
    st.session_state.process_error = None
    st.session_state.uploaded_name = uploaded_file.name

try:
    validate_pdf_file(
        pdf_bytes = pdf_bytes,
        filename  = uploaded_file.name,
        max_mb    = sidebar_cfg.get("max_pages", _cfg.max_file_size_mb),
    )
except EmptyFileError as e:
    st.error(f"📭 **Empty file:** {e}", icon="❌")
    st.stop()
except FileTooLargeError as e:
    st.error(f"📦 **File too large:** {e}", icon="❌")
    st.info(
        "**Tip:** You can split large PDFs using "
        "[Smallpdf](https://smallpdf.com/split-pdf) or "
        "[Adobe Acrobat](https://www.adobe.com/acrobat/online/split-pdf.html) — free tools.",
        icon="💡",
    )
    st.stop()
except InvalidFileTypeError as e:
    st.error(f"📄 **Invalid file:** {e}", icon="❌")
    st.stop()




from app.utils.cache import cached_extract_pdf, make_pdf_cache_key

pdf_hash = make_pdf_cache_key(pdf_bytes)

with st.spinner("📖 Reading PDF…"):
    try:
        extraction_dict = cached_extract_pdf(
            _pdf_hash = pdf_hash,
            pdf_bytes = pdf_bytes,
            filename  = uploaded_file.name,
            max_pages = sidebar_cfg["max_pages"],
        )
    except ScannedPDFError as e:
        st.error(f"🖼️ **Scanned PDF:** {e}", icon="❌")
        st.info(
            "**How to fix:** Use an online OCR tool to convert scanned pages to text:\n"
            "- [Adobe Acrobat Online](https://www.adobe.com/acrobat/online/pdf-to-word.html)\n"
            "- [Smallpdf](https://smallpdf.com/pdf-to-word)\n"
            "- [ilovepdf](https://www.ilovepdf.com/pdf_to_word)",
            icon="💡",
        )
        st.stop()
    except EmptyPDFError as e:
        st.error(f"📭 **Empty PDF:** {e}", icon="❌")
        st.stop()
    except PDFExtractionError as e:
        st.error(f"⚠️ **PDF Error:** {e}", icon="❌")
        st.stop()
    except Exception as e:
        st.error(f"❌ **Unexpected error reading PDF:** {e}", icon="❌")
        with st.expander("🔧 Error details"):
            import traceback
            st.code(traceback.format_exc(), language="python")
        st.stop()

for w in extraction_dict.get("warnings", []):
    st.warning(w, icon="⚠️")

class _MockExtraction:
    """Lightweight struct to pass extraction stats to render_pdf_info."""
    def __init__(self, d: dict):
        from app.core.pdf_extractor import PageText
        self.total_pages      = d["total_pages"]
        self.text_page_count  = d["text_page_count"]
        self.full_text        = d["full_text"]
        self.warnings         = d.get("warnings", [])
        self.meaningful_pages = [
            PageText(page_number=p["page_number"], text=p["text"])
            for p in d.get("pages", [])
            if len(p["text"]) >= 20
        ]

extraction = _MockExtraction(extraction_dict)

st.markdown('<hr class="fc-divider">', unsafe_allow_html=True)
render_pdf_info(extraction, uploaded_file.name)




st.markdown('<hr class="fc-divider">', unsafe_allow_html=True)

_active_key = groq_key if llm_provider == "groq" else google_key
if not _active_key:
    st.warning(
        "**AI API key required** to extract and verify claims.  \n"
        "Add a **Groq key** (free, no billing) in the sidebar to continue.",
        icon="🔑",
    )
    st.link_button("Get a free Groq key →", "https://console.groq.com/keys")
    st.stop()

run_col, info_col = st.columns([1, 3])

with run_col:
    run_btn = st.button(
        "🚀 Run Fact-Check",
        type                = "primary",
        use_container_width = True,
        disabled            = st.session_state.is_processing,
    )

with info_col:
    n_claims   = sidebar_cfg["max_claims"]
    est_lo     = n_claims * 5
    est_hi     = n_claims * 8
    search_src = "🌐 Tavily" if tavily_key else "🦆 DuckDuckGo"
    cache_note = "⚡ Results cached — re-runs are instant" if st.session_state.run_count > 0 else ""
    st.markdown(
        f'<div style="padding-top:8px; color:#64748b; font-size:0.85rem;">'
        f'⏱️ Est. <strong style="color:#94a3b8">{est_lo}–{est_hi}s</strong> · '
        f'Up to <strong style="color:#94a3b8">{n_claims} claims</strong> · '
        f'{search_src} · {cache_note}'
        f'</div>',
        unsafe_allow_html=True,
    )


 

if run_btn:
    st.session_state.is_processing = True
    st.session_state.result        = None
    st.session_state.process_error = None

    tracker = ProgressTracker()
    tracker.start()

    try:
        pipeline_cfg = PipelineConfig(
            llm_provider         = llm_provider,
            groq_api_key         = groq_key or None,
            google_api_key       = google_key or None,
            tavily_api_key       = tavily_key or None,
            max_pdf_pages        = sidebar_cfg["max_pages"],
            max_claims           = sidebar_cfg["max_claims"],
            inter_claim_delay    = 1.0,
            llm_timeout_secs     = 30,
            max_file_size_mb     = _cfg.max_file_size_mb,
            enable_cache         = True,
            enable_sanitiser     = True,
        )

        pipeline = FactCheckPipeline(pipeline_cfg)
        result   = pipeline.run(
            pdf_bytes   = pdf_bytes,
            filename    = uploaded_file.name,
            progress_cb = tracker.update,
        )

        st.session_state.result    = result
        st.session_state.run_count += 1

        for w in pipeline.status.warnings:
            st.warning(w, icon="⚠️")
        for e in pipeline.status.errors:
            st.error(e, icon="⚠️")

        tracker.done()
        elapsed = round(pipeline.status.elapsed_secs, 1)
        st.success(
            f"✅ Fact-checked **{result.total_claims} claims** in **{elapsed}s**.",
            icon="🎉",
        )

    except EnvironmentError as e:
        tracker.clear()
        st.error(f"🔑 **API Key Error:** {e}", icon="❌")
        st.session_state.process_error = str(e)

    except (ScannedPDFError, EmptyPDFError, PDFExtractionError) as e:
        tracker.clear()
        st.error(f"📄 **PDF Error:** {e}", icon="❌")

    except Exception as e:
        tracker.clear()
        st.error(f"❌ **Pipeline error:** {e}", icon="❌")
        st.session_state.process_error = str(e)
        with st.expander("🔧 Full error details"):
            import traceback
            st.code(traceback.format_exc(), language="python")

    finally:
        st.session_state.is_processing = False


 

result: FactCheckResult | None = st.session_state.result

if result is None:
    st.stop()

if result.total_claims == 0:
    st.warning(
        "No verifiable factual claims were found in this document.\n\n"
        "**Try:** Documents with specific statistics, dates, or factual assertions work best.",
        icon="🤷",
    )
    st.stop()

st.markdown('<hr class="fc-divider">', unsafe_allow_html=True)
st.markdown("## 📊 Results Dashboard")

render_summary_metrics(result)
st.markdown('<hr class="fc-divider">', unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.markdown("##### Verdict Distribution")
    render_verdict_chart(result)
with chart_col2:
    st.markdown("##### Claims by Category")
    render_category_chart(result)

st.markdown('<hr class="fc-divider">', unsafe_allow_html=True)

tab_cards, tab_table, tab_export = st.tabs([
    "📋 Claim Cards",
    "📊 Table View",
    "💾 Export",
])

with tab_cards:
    st.caption("Click any claim card to expand evidence and AI reasoning.")
    render_claims_detailed(result)

with tab_table:
    st.caption("Filterable, sortable table of all claims.")
    render_claims_table(result)

with tab_export:
    st.markdown("### 💾 Download Results")
    render_export_panel(result)
    with st.expander("📄 View extracted PDF text"):
        for page in extraction.meaningful_pages[:5]:
            st.markdown(f"**Page {page.page_number}**")
            st.text(page.text[:600] + ("…" if len(page.text) > 600 else ""))
            st.divider()
        if extraction.text_page_count > 5:
            st.caption(f"Showing first 5 of {extraction.text_page_count} pages.")
