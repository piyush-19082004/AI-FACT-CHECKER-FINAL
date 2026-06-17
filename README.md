# 🔍 AI Fact-Checker

> **Upload any PDF → automatically extract factual claims → verify them against live web sources → get verdicts with evidence.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-218%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Features

| Feature                       | Detail                                                                       |
| ----------------------------- | ---------------------------------------------------------------------------- |
| 📄 **PDF Upload**             | Text-based PDFs up to 50 MB / 50 pages                                       |
| 🧠 **Claim Extraction**       | Gemini 1.5 Flash extracts all verifiable factual claims, deduplicated        |
| 🌐 **Live Web Search**        | Tavily API (+ DuckDuckGo fallback) retrieves real-time evidence per claim    |
| ⚖️ **5-Class Verdict Engine** | ✅ Verified · ⚠️ Inaccurate · ❌ False · 🕐 Outdated · ❓ Unverifiable       |
| 📊 **Results Dashboard**      | Donut chart · category breakdown · colour-coded claim cards                  |
| 💾 **Export**                 | Download results as CSV or rich JSON                                         |
| 🛡️ **Hardened**               | File validation · prompt injection guard · retry+backoff · per-stage caching |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-fact-checker.git
```

### 2. Create and activate a virtual environment

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in:
#   GOOGLE_API_KEY=AIza...
#   TAVILY_API_KEY=tvly-...   (optional — DuckDuckGo fallback works without it)
```

### 5. Run the app

```bash
streamlit run app/main.py
```

Open **http://localhost:8501** — upload a PDF and click **Run Fact-Check** 🎉

---

## 🔑 API Keys

| API               | Purpose                              | Free Tier                  | Get Key                                                       |
| ----------------- | ------------------------------------ | -------------------------- | ------------------------------------------------------------- |
| **Google Gemini** | Claim extraction + verdict reasoning | 15 req/min · 1M tokens/day | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| **Tavily Search** | Live web evidence retrieval          | 1,000 queries/month        | [app.tavily.com](https://app.tavily.com)                      |

> Tavily is **optional** — the app automatically falls back to DuckDuckGo if the key is absent.

---

## ☁️ Deploy to Streamlit Cloud

Full instructions in [DEPLOYMENT.md](DEPLOYMENT.md). Quick version:

1. **Push to GitHub** (`git push`)
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → New app
3. Set **main file path** → `app/main.py`
4. Add secrets in the Streamlit Cloud dashboard:

   ```toml
   TAVILY_API_KEY = "tvly-..."
   ```

5. Click **Deploy** ✅

> See [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) for the full secrets template.

---

## 🧪 Running Tests

```bash
# All 218 tests — no API key or network required (all mocked)
pytest tests/ -v

# Single test module
pytest tests/test_fact_verifier.py -v

# With coverage
pytest tests/ --cov=app --cov-report=term-missing
```

**Test suite breakdown:**

| File                      | Tests | Coverage area                          |
| ------------------------- | ----- | -------------------------------------- |
| `test_models.py`          | 30    | Pydantic models, validation            |
| `test_pdf_extractor.py`   | 18    | PyMuPDF extraction, scanned detection  |
| `test_claim_extractor.py` | 45    | Claim parsing, dedup, chunking         |
| `test_web_searcher.py`    | 27    | Tavily, DuckDuckGo, fallback logic     |
| `test_fact_verifier.py`   | 38    | Verdict engine, pipeline orchestration |
| `test_hardening.py`       | 60    | Security, caching, rate limiting       |

---

## 🏗️ Architecture

```
[Stage 1] PDF Extractor     — PyMuPDF, header/footer dedup
   ▼
[Stage 2] Content Sanitiser — prompt injection guard (6 patterns)
   │
   ▼
   │
   ▼
[Stage 6] Result Assembler  — FactCheckResult → Streamlit UI
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design document including caching strategy, security layers, error handling, and scaling considerations.

---

## 📂 Project Structure

```
ai-fact-checker/
├── app/
│   ├── config.py                  # pydantic-settings config
│   ├── core/
│   │   ├── pdf_extractor.py       # PyMuPDF text extraction
│   │   ├── claim_extractor.py     # Gemini Flash claim detection
│   │   ├── web_searcher.py        # Tavily / DuckDuckGo search
│   │   ├── fact_verifier.py       # Gemini Pro verification
│   │   └── pipeline.py            # Pipeline orchestrator
│   ├── models/
│   │   ├── claim.py               # Claim, ClaimCategory, FactCheckResult
│   │   └── verdict.py             # Verdict, VerdictLabel, EvidenceSource
│   └── utils/
│       ├── cache.py               # @st.cache_data + SHA-256 cache keys
│       ├── logger.py              # structlog structured logging
│       ├── rate_limiter.py        # Retry, timeout, rate-limit decorators
│       └── validators.py          # File, content, API-key validation
├── tests/                         # 218 tests — all mocked, no API key needed
├── .streamlit/
│   ├── config.toml                # Dark theme, upload size limits
│   └── secrets.toml.example       # Secrets template (copy → secrets.toml locally)
├── .env.example                   # Local dev environment template
├── .gitignore                     # Excludes .env, secrets.toml, .venv, PDFs
├── requirements.txt               # Pinned versions for reproducible deploys
├── ARCHITECTURE.md                # Full design document
├── DEPLOYMENT.md                  # Step-by-step deploy checklist
└── README.md                      # This file
```

---

## 🛡️ Security Notes

- API keys are **never** stored in code — use `.env` locally, Streamlit Cloud Secrets in production
- `.env` and `secrets.toml` are in `.gitignore` and will **never** be committed
- Uploaded PDFs are held in **session memory only** — never persisted to disk
- PDF text is **sanitised** before injection into LLM prompts (6 injection patterns)
- File uploads are validated for **magic bytes** (`%PDF-`) and **size limits** before processing
- Users are informed that content is sent to Gemini and Tavily APIs

---

## 📋 Development Roadmap

- [x] **Phase 1** — Foundation (models, PDF extraction, 48 tests)
- [x] **Phase 2** — Claim extraction with Gemini + LangChain (45 tests)
- [x] **Phase 3** — Web search + fact verification (65 tests)
- [x] **Phase 4** — Full Streamlit UI (charts, export, claim cards)
- [x] **Phase 5** — Hardening (file validation, prompt injection guard, caching, retry, timeout)
- [x] **Phase 6** — Deployment (frozen requirements, README, ARCHITECTURE.md, DEPLOYMENT.md)

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Add tests for your changes (`pytest tests/ -v`)
4. Open a pull request

---

## 📄 License

MIT © 2024 — See [LICENSE](LICENSE) for details.

---

_Built with ❤️ using Streamlit · LangChain · Gemini 1.5 · Tavily · PyMuPDF_
