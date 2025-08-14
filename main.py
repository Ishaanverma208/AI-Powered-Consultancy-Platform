# backend/main.py
import os
import time
import uuid
import json
from typing import Dict, Any, List, Optional, Literal
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from jinja2 import Template

# -------------------------
# Load env
# -------------------------
load_dotenv()

# Provider config
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "ollama").lower()  # 'ollama' | 'gemini' | 'groq'
# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile").strip()
# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1").strip()

# Search
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "").strip()

# -------------------------
# App
# -------------------------
app = FastAPI(title="SmartConsult – Multi-Provider (Ollama / Gemini / Groq)")

# In-memory store (MVP)
CASES: Dict[str, Dict[str, Any]] = {}

# -------------------------
# Pydantic models
# -------------------------
class StartCaseReq(BaseModel):
    company_name: str
    industry: str = ""
    region: str = ""
    size: str = ""
    revenue: str = ""
    users: str = ""
    problem_statement: str
    provider: Optional[Literal["ollama", "gemini", "groq"]] = None  # optional override

class FollowupAnswers(BaseModel):
    case_id: str
    answers: Dict[str, str]
    provider: Optional[Literal["ollama", "gemini", "groq"]] = None  # optional override

class AskReq(BaseModel):
    prompt: str
    provider: Optional[Literal["ollama", "gemini", "groq"]] = None

# -------------------------
# Provider helpers
# -------------------------
def _choose_provider(explicit: Optional[str]) -> str:
    if explicit in {"ollama", "gemini", "groq"}:
        return explicit
    return DEFAULT_PROVIDER

def generate_with_gemini(prompt: str, model: str = None, api_key: str = None, max_tokens: int = 1024, temperature: float = 0.2) -> str:
    model = (model or GEMINI_MODEL) or "gemini-1.5-flash"
    api_key = api_key or GEMINI_API_KEY
    if not api_key:
        return "[ERROR] GEMINI_API_KEY missing."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }
    try:
        r = requests.post(url, params=params, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        # Extract text from candidates[0].content.parts[*].text
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            texts: List[str] = []
            for p in parts:
                if isinstance(p, dict) and "text" in p:
                    texts.append(p["text"])
                elif isinstance(p, str):
                    texts.append(p)
            return "".join(texts).strip() or json.dumps(data)[:4000]
        return json.dumps(data)[:4000]
    except requests.exceptions.HTTPError:
        return f"[ERROR] Gemini HTTP {r.status_code}: {r.text}"
    except Exception as e:
        return f"[ERROR] Gemini call failed: {str(e)}"

def generate_with_groq(prompt: str, model: str = None, api_key: str = None, max_tokens: int = 1024, temperature: float = 0.2) -> str:
    model = (model or GROQ_MODEL) or "llama-3.1-70b-versatile"
    api_key = api_key or GROQ_API_KEY
    if not api_key:
        return "[ERROR] GROQ_API_KEY missing."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a senior management consultant. Be concise, structured, and actionable."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return (text or json.dumps(data)[:4000]).strip()
    except requests.exceptions.HTTPError:
        return f"[ERROR] Groq HTTP {r.status_code}: {r.text}"
    except Exception as e:
        return f"[ERROR] Groq call failed: {str(e)}"

def generate_with_ollama(prompt: str, model: str = None, host: str = None, max_tokens: int = 1024, temperature: float = 0.2) -> str:
    model = (model or OLLAMA_MODEL) or "llama3.1"
    host = host or OLLAMA_HOST
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        },
        "stream": False
    }
    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or json.dumps(data)[:4000]).strip()
    except requests.exceptions.ConnectionError:
        return "[ERROR] Ollama server not reachable. Is Ollama running?"
    except requests.exceptions.HTTPError:
        return f"[ERROR] Ollama HTTP {r.status_code}: {r.text}"
    except Exception as e:
        return f"[ERROR] Ollama call failed: {str(e)}"

def llm_generate(prompt: str, provider: Optional[str] = None) -> str:
    p = _choose_provider(provider)
    if p == "gemini":
        return generate_with_gemini(prompt)
    if p == "groq":
        return generate_with_groq(prompt)
    # default to ollama
    return generate_with_ollama(prompt)

# -------------------------
# SerpAPI helpers (competitors / case-studies)
# -------------------------
def serpapi_search(query: str, num: int = 6):
    if not SERPAPI_KEY:
        return []
    url = "https://serpapi.com/search.json"
    params = {"q": query, "api_key": SERPAPI_KEY, "num": num}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        items = []
        for o in data.get("organic_results", [])[:num]:
            items.append({
                "title": o.get("title"),
                "link": o.get("link"),
                "snippet": o.get("snippet")
            })
        return items
    except Exception:
        return []

def find_competitors(company_name: str, industry: str = "", region: str = ""):
    queries = [
        f"competitors of {company_name} {industry} {region}".strip(),
        f"companies similar to {company_name} {industry}".strip(),
        f"top competitors {company_name}".strip()
    ]
    results = []
    for q in queries:
        if len(results) >= 6:
            break
        hits = serpapi_search(q, num=6)
        for h in hits:
            if not any(h.get("link") == r.get("link") or h.get("title") == r.get("title") for r in results):
                results.append(h)
                if len(results) >= 6:
                    break
    return results

def fetch_case_studies(topic: str, num: int = 6):
    q = f'{topic} "case study" site:medium.com OR site:forbes.com OR site:techcrunch.com'
    return serpapi_search(q, num=num)

# -------------------------
# Report generation
# -------------------------
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

REPORT_TEMPLATE = Template("""
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Consulting Report - {{company}}</title></head>
<body style="font-family:Arial,Helvetica,sans-serif;line-height:1.4">
  <h1>Consulting Report — {{company}}</h1>
  <h3>Problem</h3><p>{{problem}}</p>

  <h3>Identified Competitors</h3>
  <ul>
  {% for c in competitors %}
    <li><b>{{c.title}}</b> — {{c.snippet}} {% if c.link %}<a href="{{c.link}}">source</a>{% endif %}</li>
  {% endfor %}
  </ul>

  <h3>Follow-up Q&A</h3>
  <ul>
  {% for q,a in qna.items() %}
    <li><b>{{q}}</b><br/>Answer: {{a}}</li>
  {% endfor %}
  </ul>

  <h3>Peer Case Studies</h3>
  <ul>
  {% for s in case_studies %}
    <li><a href="{{s.link}}">{{s.title}}</a> — {{s.snippet}}</li>
  {% endfor %}
  </ul>

  <h3>Recommendation</h3>
  <pre style="white-space:pre-wrap;">{{recommendation}}</pre>
</body>
</html>
""")

def save_report_html(case_obj: Dict[str, Any]) -> str:
    html = REPORT_TEMPLATE.render(
        company=case_obj["company_name"],
        problem=case_obj["problem_statement"],
        competitors=case_obj.get("competitors", []),
        qna=case_obj.get("followup_answers", {}),
        case_studies=case_obj.get("case_studies", []),
        recommendation=json.dumps(case_obj.get("final_recommendation", {}), indent=2)
            if isinstance(case_obj.get("final_recommendation"), dict)
            else str(case_obj.get("final_recommendation"))
    )
    filename = f"report_{case_obj['case_id']}.html"
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path

# -------------------------
# API endpoints
# -------------------------
@app.get("/")
def root():
    return {"status": "ok",
            "message": "SmartConsult backend (Ollama/Gemini/Groq). Use /docs to try.",
            "default_provider": DEFAULT_PROVIDER}

@app.post("/ask")
def ask(req: AskReq):
    out = llm_generate(req.prompt, req.provider)
    return {"provider": _choose_provider(req.provider), "output": out}

@app.post("/start_case")
def start_case(payload: StartCaseReq):
    provider = _choose_provider(payload.provider)
    case_id = str(uuid.uuid4())

    CASES[case_id] = {
        "case_id": case_id,
        "company_name": payload.company_name,
        "industry": payload.industry,
        "region": payload.region,
        "size": payload.size,
        "revenue": payload.revenue,
        "users": payload.users,
        "problem_statement": payload.problem_statement,
        "created_at": time.time(),
        "provider": provider
    }

    # 1) Competitor discovery
    competitors = find_competitors(payload.company_name, payload.industry, payload.region)
    CASES[case_id]["competitors"] = competitors

    # 2) Ask provider for follow-up questions
    comp_names = ", ".join([c.get("title","") for c in competitors]) if competitors else "none found"
    prompt = (
        f"You are a senior management consultant. A client company '{payload.company_name}' in industry '{payload.industry}' "
        f"submitted this problem: \"\"\"{payload.problem_statement}\"\"\". Competitors found: {comp_names}.\n\n"
        "Produce 4 concise, high-value follow-up questions that will help diagnose the root causes. "
        "Number them or put each on a new line."
    )
    raw = llm_generate(prompt, provider)
    questions = [line.strip("0123456789. -").strip() for line in raw.splitlines() if line.strip()]
    questions = questions[:6] if questions else [raw]  # if error, surface raw
    CASES[case_id]["followup_questions"] = questions

    return {"case_id": case_id, "provider": provider, "followup_questions": questions, "competitors": competitors}

@app.post("/submit_answers")
def submit_answers(payload: FollowupAnswers):
    case = CASES.get(payload.case_id)
    if not case:
        raise HTTPException(status_code=404, detail="case not found")

    provider = _choose_provider(payload.provider or case.get("provider"))
    case["followup_answers"] = payload.answers

    # 1) Case studies
    topic = case["problem_statement"]
    case_studies = fetch_case_studies(topic, num=6)
    case["case_studies"] = case_studies

    # 2) Ask LLM for structured recommendation (force JSON)
    prompt_parts = [
        "You are a senior management consultant. Produce a structured consulting note.",
        f"Client: {case['company_name']}",
        f"Industry: {case['industry']}",
        f"Problem: \"\"\"{case['problem_statement']}\"\"\"",
        "Follow-up Q&A:"
    ]
    for q, a in case["followup_answers"].items():
        prompt_parts.append(f"Q: {q}\nA: {a}")
    if case_studies:
        prompt_parts.append("\nRelevant case studies (title + snippet):")
        for cs in case_studies:
            prompt_parts.append(f"- {cs.get('title')} : {cs.get('snippet')}")
    prompt_parts.append(
        "\nReturn ONLY valid minified JSON with keys:\n"
        '{"executive_summary":"...","diagnosis":["..."],'
        '"plan_30_60_90":{"30":["..."],"60":["..."],"90":["..."]},'
        '"metrics":["..."],"quick_wins":["..."]}'
    )
    prompt = "\n\n".join(prompt_parts)
    raw = llm_generate(prompt, provider)

    # 3) Parse JSON robustly
    rec: Dict[str, Any]
    try:
        rec = json.loads(raw)
    except Exception:
        start = raw.find("{"); end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                rec = json.loads(raw[start:end+1])
            except Exception:
                rec = {"raw": raw}
        else:
            rec = {"raw": raw}

    case["final_recommendation"] = rec

    # 4) Save HTML report
    html_path = save_report_html(case)
    case["report_html"] = html_path
    case["report_pdf"] = None  # (optional) add wkhtmltopdf/pdfkit if you want

    return {
        "case_id": payload.case_id,
        "provider": provider,
        "recommendation": rec,
        "report_html": html_path
    }

@app.get("/report/{case_id}")
def get_report(case_id: str):
    case = CASES.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return {
        "case_id": case_id,
        "provider": case.get("provider"),
        "report_html": case.get("report_html"),
        "report_pdf": case.get("report_pdf"),
        "recommendation": case.get("final_recommendation")
    }
