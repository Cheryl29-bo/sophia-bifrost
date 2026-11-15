import json, os
from datetime import date
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
app = FastAPI(title="Agent Sophia x Bifrost x Tavily")

TAVILY_KEY = os.getenv("TAVILY_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
# Every LLM call goes through the Bifrost gateway.
llm = OpenAI(base_url=os.getenv("LLM_BASE_URL", "http://localhost:8080/openai"), api_key="dummy")


class Req(BaseModel):
    profile: str
    role: str
    location: str = "Remote / Worldwide"


def ask(system: str, user: str) -> str:
    r = llm.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return r.choices[0].message.content


def clean_json(t: str):
    return json.loads(t.replace("```json", "").replace("```", "").strip())


@app.get("/health")
def health():
    try:
        ask("Reply with OK", "ping")
        return {"gateway": "bifrost ok", "tavily_key": bool(TAVILY_KEY)}
    except Exception as e:
        return {"gateway": f"error: {e}", "tavily_key": bool(TAVILY_KEY)}


@app.post("/api/match")
async def match(req: Req):
    # 1. Job discovery with Tavily
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {TAVILY_KEY}"},
            json={"query": f"{req.role} job openings {req.location}", "max_results": 6},
        )
    jobs = [
        {"title": j["title"], "url": j["url"], "snippet": j["content"][:400]}
        for j in r.json().get("results", [])
    ]
    # 2. Matching, skill gaps, copilot: one structured call through Bifrost
    system = (
        "You are Sophia, a career agent. Assess each listing against the candidate using ONLY the listing text; "
        "never invent employers, deadlines or requirements. Separate mandatory from preferred requirements. "
        "Treat missing candidate info as UNKNOWN, not as failure. A failed mandatory requirement means Not eligible "
        "regardless of score. Do not infer protected characteristics. Return ONLY JSON: "
        '{"matches":[{"index":0,"eligibility_status":"Likely eligible|Potential match|Not eligible",'
        '"fit_score":0-100,"confidence":"low|medium|high","mandatory":[""],"preferred":[""],'
        '"matched_criteria":[""],"missing_criteria":[""],"unknown":[""],"explanation":"",'
        '"next_step":"","cover_note":""}]} '
        "Weights for fit_score: skills 30, experience 25, education 20, domain 10, preferences 10, other 5. "
        "If the listing text is thin, set confidence to low."
    )
    user = f"Candidate profile:\n{req.profile}\n\nListings:\n{json.dumps(jobs)}"
    try:
        out = clean_json(ask(system, user))["matches"]
        for m in out:
            m["job"] = jobs[m["index"]]
        order = {"Likely eligible": 0, "Potential match": 1, "Not eligible": 2}
        out.sort(key=lambda m: (order.get(m["eligibility_status"], 3), -m["fit_score"]))
        return {"results": out, "checked": date.today().isoformat()}
    except Exception as e:
        return {"results": [], "raw_jobs": jobs, "error": str(e)}


PAGE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Agent Sophia</title>
<style>body{font-family:system-ui;max-width:760px;margin:20px auto;padding:0 14px;color:#0b1f3a}
h1{color:#0b1f3a}textarea,input{width:100%;padding:10px;margin:6px 0;border:1px solid #ccd;border-radius:8px;box-sizing:border-box}
button{background:#0b1f3a;color:#fff;border:0;padding:12px 18px;border-radius:8px;font-size:16px}
.card{border:1px solid #dde;border-radius:10px;padding:12px;margin:12px 0}.s{font-weight:700;color:#0a7}</style>
<h1>Agent Sophia</h1><p>AI career agent. Jobs via Tavily, intelligence via Bifrost.</p>
<input id=role placeholder="Target role e.g. Product Manager">
<input id=loc placeholder="Location e.g. Remote">
<textarea id=profile rows=5 placeholder="Paste your short profile or CV summary"></textarea>
<button onclick=go()>Find and match jobs</button><div id=out></div>
<script>
async function go(){const o=document.getElementById('out');o.textContent='Searching and matching...';
const r=await fetch('/api/match',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({role:role.value,location:loc.value||'Remote',profile:profile.value})});
const d=await r.json();if(!d.results.length){o.textContent=d.error||'No results';return}
o.innerHTML=d.results.map(m=>`<div class=card><b><a href="${m.job.url}" target=_blank>${m.job.title}</a></b>
<div><span class=s>${m.eligibility_status}</span> | Fit ${m.fit_score}% | Confidence: ${m.confidence}</div>
<p>${m.explanation}</p><p><b>Matched:</b> ${m.matched_criteria.join('; ')||'-'}</p>
<p><b>Missing:</b> ${m.missing_criteria.join('; ')||'-'}</p><p><b>Unconfirmed:</b> ${(m.unknown||[]).join('; ')||'-'}</p>
<p><b>Next step:</b> ${m.next_step}</p><p><b>Draft cover note (review before use):</b> ${m.cover_note}</p>
<small>Source: ${m.job.url} | Last checked: ${d.checked} | Live search result, verify with the employer.</small></div>`).join('')}
</script>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
