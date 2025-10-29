"""
RIS Minimal Open Code — FastAPI MVP (no‑login)
Date: 2025‑10‑28
Author: Masoud Moghaddam • Advisor: GPT‑5 Thinking

Overview
- Public, no‑login API implementing the no‑code spec (Sections 3, 22–24) in a minimal, single‑file FastAPI app.
- Domains: security | economy | health (homogenized via DHAL → CFV: I, V, C, A, E)
- Endpoints:
  POST /pulse                 -> ingest Pulse Frame and run a Δt cycle (MVP)
  GET  /alerts                -> list alerts
  GET  /recommendations       -> list recommendations
  GET  /snapshot              -> latest snapshot (saliency_topK, reinforced_paths_topK, homeostasis)
  GET  /telemetry             -> KPIs + WDC actions + agent reputations (simplified)
  GET  /config                -> active domain configs

Run
  uvicorn app:app --reload --port 8000

Notes
- This is an intentionally compact MVP: in‑memory stores, simplified spherical grid (logical sectors),
  shadow→main promotion simulated, basic adaptive control hooks.
- No authentication, no PII, advisory‑only outputs (no actuation endpoints).
"""

from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any, Tuple
from datetime import datetime, timezone
import math
import uuid
import statistics

# ----------------------------
# Domain & DHAL Config Profiles
# ----------------------------
Domain = Literal["security", "economy", "health"]

class ControlParams(BaseModel):
    tau: float        # saliency threshold τ
    kappa: int        # consensus κ
    gamma: float      # update throttle Γ
    lambda_: float = Field(alias="lambda")  # ← نام فیلد امن در پایتون + alias برای JSON
    M: int            # max concurrent missions (simplified)
    # در Pydantic v2:

    model_config = ConfigDict(populate_by_name=True)  # اجازه مقداردهی با نام فیلد همزمان با alias

class DHALWeights(BaseModel):
    I: float
    V: float
    C: float
    A: float
    E: float

class DomainProfile(BaseModel):
    name: Domain
    controls: ControlParams
    dhal: DHALWeights
    policy_tier: int  # 1,2,3 (health/defense stricter)

DEFAULT_PROFILES: Dict[Domain, DomainProfile] = {
    "security": DomainProfile(
        name="security",
        controls=ControlParams(tau=0.68, kappa=4, gamma=0.15, lambda_=0.05, M=8),
        dhal=DHALWeights(I=0.25, V=0.20, C=0.15, A=0.25, E=0.15),
        policy_tier=2,
    ),
    "economy": DomainProfile(
        name="economy",
        controls=ControlParams(tau=0.68, kappa=4, gamma=0.18, lambda_=0.05, M=9),
        dhal=DHALWeights(I=0.25, V=0.20, C=0.25, A=0.20, E=0.10),
        policy_tier=3,
    ),
    "health": DomainProfile(
        name="health",
        controls=ControlParams(tau=0.70, kappa=5, gamma=0.14, lambda_=0.05, M=6),
        dhal=DHALWeights(I=0.25, V=0.20, C=0.15, A=0.30, E=0.10),
        policy_tier=1,
    ),
}


# ----------------------------
# Data Contracts (Pydantic)
# ----------------------------
class SourceRef(BaseModel):
    source_id: str
    type: Literal["text", "audio", "sensor", "event", "price", "netflow", "generic"] = "generic"
    quality: float = Field(ge=0.0, le=1.0, default=1.0)

class PulseFrame(BaseModel):
    pulse_id: Optional[str] = None
    time_window: Dict[str, Any] = Field(default_factory=lambda: {
        "t_start": datetime.now(timezone.utc).isoformat(),
        "t_end": datetime.now(timezone.utc).isoformat(),
        "timezone": "UTC"
    })
    domain: Domain
    sources: List[SourceRef]
    payload: Dict[str, Any] = Field(default_factory=dict)  # domain‑specific raw features
    latency_budget_ms: int = 1500

# Tickets & Reports (simplified MVP)
class InterruptTicket(BaseModel):
    ticket_id: str
    coords: str  # sector id (logical)
    saliency: float
    priority: Literal["critical", "high", "normal"] = "high"
    ttl_ms: int = 3000
    reason_codes: List[str] = []

class SpecialistReport(BaseModel):
    report_id: str
    ticket_id: str
    coords: str
    delta_signature: Dict[str, float]  # compact deltas
    hypothesis: Literal["transient", "step-change", "adversarial"]
    confidence: float
    cost_ms: int
    agent_id: str
    agent_reputation: float

# Outputs
class Alert(BaseModel):
    alert_id: str
    domain: Domain
    type: str
    coords: str
    when: str
    why: str
    confidence: float
    severity: Literal["Low", "Medium", "High"]
    next_step: str
    ttl_ms: int

class Recommendation(BaseModel):
    rec_id: str
    action: str
    preconditions: List[str]
    expected_impact: str
    rollback_plan: str
    explainability: str

class Snapshot(BaseModel):
    saliency_topK: List[Tuple[str, float]]
    reinforced_paths_topK: List[str]
    homeostasis_status: str

class Telemetry(BaseModel):
    KPIs: Dict[str, float]
    wdc_actions: List[str]
    agent_reputations: Dict[str, float]

# ----------------------------
# In‑Memory State (MVP)
# ----------------------------



import json, os, logging

CONFIG_DIR = "config"

def load_domain_profiles():
    profiles = {}
    for file in os.listdir(CONFIG_DIR):
        if not file.endswith(".json"):
            continue
        path = os.path.join(CONFIG_DIR, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                domain = data["domain"]
                profiles[domain] = DomainProfile(
                    name=domain,
                    controls = ControlParams.model_validate(data["controls"]),
                    dhal=DHALWeights(**data["dhal_weights"]),
                    policy_tier=data["policy_tier"]
                )
            logging.info(f"Loaded config for domain: {domain}")
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")
    return profiles








# ----------------------------
# In‑Memory State (MVP)
# ----------------------------
class SphereState:
    def __init__(self):
        self.domain_profiles: Dict[Domain, DomainProfile] = DEFAULT_PROFILES.copy()
        self.active_profile: Domain = "security"  # default; can be set per pulse
        # logical sectors: 48 sectors over sphere (simplified)
        self.sectors = {f"S{i}": {"sal": 0.0, "weight": 1.0} for i in range(48)}
        self.shadow_updates: List[Tuple[str, float]] = []
        self.main_paths: Dict[str, float] = {}  # reinforced weights per sector path
        self.alerts: List[Alert] = []
        self.recs: List[Recommendation] = []
        self.agent_rep: Dict[str, float] = {}
        self.kpi: Dict[str, float] = {
            "precision": 0.0,
            "recall": 0.0,
            "fir": 0.0,
            "mtti": 1.5,
            "oscillation": 0.0,
        }
        self.wdc_actions: List[str] = []
        self.tickets_open: Dict[str, InterruptTicket] = {}

    # simple helper: map a key to a sector id (deterministic)
    def sector_of(self, key: str) -> str:
        h = abs(hash(key)) % len(self.sectors)
        return f"S{h}"

STATE = SphereState()
STATE.domain_profiles = load_domain_profiles()

app = FastAPI(title="RIS Minimal Open API (MVP)", version="0.1.0")

# ----------------------------
# Utilities — DHAL → CFV & Saliency
# ----------------------------

def _zscore(x: float, mu: float, sigma: float) -> float:
    if sigma <= 1e-9:
        return 0.0
    return (x - mu) / sigma

# For MVP we expect caller to provide basic baseline in payload or defaults will be used
DEFAULT_BASELINES = {
    "security": {"dst_rep": 0.85, "conn_5s": 15, "err_rate": 0.01, "proc_hash_score": 0.93, "geo_anom": 0.05},
    "economy":  {"news_sentiment": 0.0, "xsec_corr": 0.4, "vol_of_vol": 0.1, "imbalance": 0.0, "spread_bps": 10.0},
    "health":   {"HR": 75.0, "RR": 16.0, "SpO2": 97.0, "Temp": 36.8, "HRV": 60.0},
}

# Map raw domain features -> CFV (I,V,C,A,E), using simple z‑scores vs baseline

def dhal_map_to_cfv(domain: Domain, payload: Dict[str, Any]) -> Dict[str, float]:
    bl = DEFAULT_BASELINES[domain]
    # crude std devs for MVP (could be estimated online)
    SD = {
        "security": {"dst_rep": 0.1, "conn_5s": 10, "err_rate": 0.02, "proc_hash_score": 0.05, "geo_anom": 0.1},
        "economy":  {"news_sentiment": 0.3, "xsec_corr": 0.2, "vol_of_vol": 0.05, "imbalance": 0.4, "spread_bps": 5.0},
        "health":   {"HR": 12.0, "RR": 4.0, "SpO2": 1.5, "Temp": 0.5, "HRV": 15.0},
    }[domain]

    if domain == "security":
        I = _zscore(float(payload.get("conn_5s", bl["conn_5s"])), bl["conn_5s"], SD["conn_5s"]) + \
            -_zscore(float(payload.get("dst_rep", bl["dst_rep"])), bl["dst_rep"], SD["dst_rep"])  # low rep increases intensity
        V = _zscore(float(payload.get("err_rate", bl["err_rate"])), bl["err_rate"], SD["err_rate"]) 
        C = _zscore(float(payload.get("geo_anom", bl["geo_anom"])), bl["geo_anom"], SD["geo_anom"]) 
        A = -_zscore(float(payload.get("proc_hash_score", bl["proc_hash_score"])), bl["proc_hash_score"], SD["proc_hash_score"]) 
        E = 0.0  # not used in MVP
    elif domain == "economy":
        I = abs(_zscore(float(payload.get("imbalance", bl["imbalance"])), bl["imbalance"], SD["imbalance"]))
        V = _zscore(float(payload.get("vol_of_vol", bl["vol_of_vol"])), bl["vol_of_vol"], SD["vol_of_vol"]) 
        C = _zscore(float(payload.get("xsec_corr", bl["xsec_corr"])), bl["xsec_corr"], SD["xsec_corr"]) 
        A = abs(_zscore(float(payload.get("spread_bps", bl["spread_bps"])), bl["spread_bps"], SD["spread_bps"]))
        E = -_zscore(float(payload.get("news_sentiment", bl["news_sentiment"])), bl["news_sentiment"], SD["news_sentiment"])  # negative news -> positive E
    else:  # health
        I = _zscore(float(payload.get("HR", bl["HR"])), bl["HR"], SD["HR"]) + _zscore(float(payload.get("RR", bl["RR"])), bl["RR"], SD["RR"]) 
        V = -_zscore(float(payload.get("HRV", bl["HRV"])), bl["HRV"], SD["HRV"])  # lower HRV -> higher volatility/stress
        C = 0.0  # omitted in MVP (needs multi‑lead coherence)
        A = -_zscore(float(payload.get("SpO2", bl["SpO2"])), bl["SpO2"], SD["SpO2"]) + abs(_zscore(float(payload.get("Temp", bl["Temp"])), bl["Temp"], SD["Temp"]))
        E = 0.0  # activity/med not modeled in MVP

    return {"I": float(I), "V": float(V), "C": float(C), "A": float(A), "E": float(E)}


def saliency_from_cfv(domain: Domain, cfv: Dict[str, float]) -> float:
    w = STATE.domain_profiles[domain].dhal
    # sigmoid squashing for 0..1
    lin = w.I * cfv["I"] + w.V * cfv["V"] + w.C * cfv["C"] + w.A * cfv["A"] + w.E * cfv["E"]
    s = 1.0 / (1.0 + math.exp(-lin))
    return max(0.0, min(1.0, s))

# ----------------------------
# Core Cycle (MVP)
# ----------------------------

def run_cycle(pulse: PulseFrame) -> Dict[str, Any]:
    profile = STATE.domain_profiles[pulse.domain]
    STATE.active_profile = pulse.domain

    # Map to sector (simple): combine domain + a key from payload
    sector_key = f"{pulse.domain}:{sorted(list(pulse.payload.keys()))[:1]}"
    sector_id = STATE.sector_of(sector_key)

    # Build CFV → Saliency
    cfv = dhal_map_to_cfv(pulse.domain, pulse.payload)
    S = saliency_from_cfv(pulse.domain, cfv)

    # Issue ticket if S > τ
    tickets: List[InterruptTicket] = []
    if S > profile.controls.tau:
        t = InterruptTicket(
            ticket_id=str(uuid.uuid4()),
            coords=sector_id,
            saliency=S,
            priority="high" if S < 0.85 else "critical",
            reason_codes=["delta_vs_baseline"],
        )
        STATE.tickets_open[t.ticket_id] = t
        tickets.append(t)

    # WDC Gate (simplified): limit to M tickets per cycle
    tickets = tickets[: profile.controls.M]

    reports: List[SpecialistReport] = []
    for t in tickets:
        agent_id = f"agent-{t.coords}"
        rep = STATE.agent_rep.get(agent_id, 0.5)
        conf = min(0.95, 0.6 + 0.5 * (t.saliency - profile.controls.tau))
        report = SpecialistReport(
            report_id=str(uuid.uuid4()),
            ticket_id=t.ticket_id,
            coords=t.coords,
            delta_signature=cfv,
            hypothesis="step-change" if t.saliency > (profile.controls.tau + 0.08) else "transient",
            confidence=conf,
            cost_ms=160,
            agent_id=agent_id,
            agent_reputation=rep,
        )
        reports.append(report)

    # Glia ensemble: require κ reports for same coords (MVP: we usually have <=1)
    promoted = False
    if reports:
        # Group by coords
        by_coords: Dict[str, List[SpecialistReport]] = {}
        for r in reports:
            by_coords.setdefault(r.coords, []).append(r)
        for coords, group in by_coords.items():
            if len(group) >= profile.controls.kappa:
                # promote path weight in shadow and maybe to main subject to Γ
                avg_conf = statistics.fmean([g.confidence for g in group])
                delta = profile.controls.gamma * avg_conf
                STATE.shadow_updates.append((coords, delta))
                promoted = True
            else:
                # MVP: still allow micro‑promotion to shadow to keep pipeline moving
                avg_conf = statistics.fmean([g.confidence for g in group])
                delta = 0.4 * profile.controls.gamma * avg_conf
                STATE.shadow_updates.append((coords, delta))

    # A/B: promote some shadow to main respecting Γ and decay λ
    for coords, delta in STATE.shadow_updates:
        cur = STATE.main_paths.get(coords, 0.0)
        STATE.main_paths[coords] = max(0.0, min(1.0, cur + delta))
    STATE.shadow_updates.clear()

    # decay
    for coords in list(STATE.main_paths.keys()):
        STATE.main_paths[coords] *= (1.0 - profile.controls.lambda_)
        if STATE.main_paths[coords] < 1e-3:
            del STATE.main_paths[coords]

    # Outputs
    alerts: List[Alert] = []
    recs: List[Recommendation] = []
    for t in tickets:
        when = datetime.now(timezone.utc).isoformat()
        sev = "High" if t.saliency > 0.8 else "Medium"
        cfv_rounded = {k: round(v, 2) for k, v in cfv.items()}
        why = f"Saliency {t.saliency:.2f} > τ {profile.controls.tau:.2f}; CFV={cfv_rounded}"
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            domain=pulse.domain,
            type="anomaly",
            coords=t.coords,
            when=when,
            why=why,
            confidence=min(0.95, 0.7 + 0.3 * (t.saliency - profile.controls.tau)),
            severity=sev,
            next_step="Investigate (HIL)" if profile.policy_tier == 1 else "Apply Guarded Advisory",
            ttl_ms=900000,
        )
        alerts.append(alert)
        STATE.alerts.append(alert)

        # Recommendation template per domain
        if pulse.domain == "security":
            rec = Recommendation(
                rec_id=str(uuid.uuid4()),
                action="Guarded egress throttle (10m)",
                preconditions=["Host criticality check", "No maintenance window"],
                expected_impact="Reduce outbound surge and observe Δ in 15m",
                rollback_plan="Auto‑rollback if Δ<0.2 for 15m",
                explainability="Elevated conn_5s, low dst_rep, errors↑",
            )
        elif pulse.domain == "economy":
            rec = Recommendation(
                rec_id=str(uuid.uuid4()),
                action="Hedge correlated exposures (light)",
                preconditions=["Correlation>0.7", "Liquidity OK"],
                expected_impact="Lower tail risk during headline shock",
                rollback_plan="Remove hedge when Δ normalizes",
                explainability="Sentiment shock + xsec_corr↑ + vol‑of‑vol↑",
            )
        else:
            rec = Recommendation(
                rec_id=str(uuid.uuid4()),
                action="Trigger guarded bedside alarm",
                preconditions=["Artifact low", "SpO2<93% OR RR>24"],
                expected_impact="Early clinical attention; reduce escalation risk",
                rollback_plan="Stop alarm if Δ<0.2 for 15m",
                explainability="SpO2↓ + RR↑ + HR↑ with rest context",
            )
        recs.append(rec)
        STATE.recs.append(rec)

    # Update simple KPIs (MVP heuristics)
    if tickets:
        # treat every ticket as TP for MVP; FIR estimate via (saliency margin)
        precision = min(0.95, 0.8 + 0.2 * (S - profile.controls.tau))
        recall = min(0.9, 0.7 + 0.3 * (S - profile.controls.tau))
        fir = max(0.0, 0.15 - 0.5 * (S - profile.controls.tau))
        mtti = max(1.5, 1.9 - 0.5 * (S - profile.controls.tau))
        STATE.kpi.update({"precision": precision, "recall": recall, "fir": fir, "mtti": mtti, "oscillation": 0.0})
        # reputation
        for r in reports:
            STATE.agent_rep[r.agent_id] = min(1.0, STATE.agent_rep.get(r.agent_id, 0.5) + 0.02)
        STATE.wdc_actions.append(f"cycle-promote:{len(tickets)}")
    else:
        # decay KPIs toward baseline
        STATE.kpi["precision"] *= 0.98
        STATE.kpi["recall"] *= 0.98
        STATE.kpi["fir"] *= 0.98
        STATE.kpi["mtti"] = 1.5

    # Snapshot
    saliency_topK = sorted([(sector_id, S)], key=lambda x: x[1], reverse=True)[:5]
    reinforced = sorted(STATE.main_paths.items(), key=lambda x: x[1], reverse=True)
    snapshot = Snapshot(
        saliency_topK=saliency_topK,
        reinforced_paths_topK=[k for k, _ in reinforced[:5]],
        homeostasis_status="stable" if STATE.kpi.get("oscillation", 0.0) < 0.5 else "oscillating",
    )

    return {
        "tickets": [t.dict() for t in tickets],
        "reports": [r.dict() for r in reports],
        "alerts": [a.dict() for a in alerts],
        "recs": [r.dict() for r in recs],
        "snapshot": snapshot.dict(),
        "kpi": STATE.kpi,
    }

# ----------------------------
# API Routes
# ----------------------------
# — بعد از app = FastAPI(...) این‌ها را بگذار —

@app.get("/", tags=["system"])
async def root():
    return {
        "service": "RIS Core Alpha",
        "version": "0.1.0",
        "endpoints": ["/docs", "/config", "/telemetry", "/alerts", "/recommendations", "/snapshot", "/pulse"]
    }

@app.get("/healthz", tags=["system"])
async def healthz():
    return {"ok": True}

@app.get("/config")
async def get_config():
    return {k: v.model_dump(by_alias=True) for k, v in STATE.domain_profiles.items()}

@app.get("/reload_config")
async def reload_config():
    STATE.domain_profiles = load_domain_profiles()
    return {"status": "ok", "domains": list(STATE.domain_profiles.keys())}

@app.post("/pulse")
async def post_pulse(pulse: PulseFrame):
    if pulse.domain not in DEFAULT_PROFILES:
        raise HTTPException(status_code=400, detail="unsupported domain")
    if not pulse.pulse_id:
        pulse.pulse_id = str(uuid.uuid4())
    result = run_cycle(pulse)
    return {"accepted": True, "pulse_id": pulse.pulse_id, **result}

@app.get("/alerts", response_model=List[Alert])
async def get_alerts(domain: Optional[Domain] = None, limit: int = Query(50, ge=1, le=500)):
    items = [a for a in STATE.alerts if (domain is None or a.domain == domain)]
    return list(reversed(items))[:limit]

@app.get("/recommendations", response_model=List[Recommendation])
async def get_recs(limit: int = Query(50, ge=1, le=500)):
    return list(reversed(STATE.recs))[:limit]

@app.get("/snapshot", response_model=Snapshot)
async def get_snapshot():
    reinforced = sorted(STATE.main_paths.items(), key=lambda x: x[1], reverse=True)
    top_sal = [(k, v["sal"]) for k, v in STATE.sectors.items()]
    # MVP uses last computed snapshot via run_cycle; if none, provide empty structure
    return Snapshot(
        saliency_topK=top_sal[:5],
        reinforced_paths_topK=[k for k, _ in reinforced[:5]],
        homeostasis_status="stable" if STATE.kpi.get("oscillation", 0.0) < 0.5 else "oscillating",
    )

@app.get("/telemetry", response_model=Telemetry)
async def get_telemetry():
    return Telemetry(KPIs=STATE.kpi, wdc_actions=STATE.wdc_actions[-20:], agent_reputations=STATE.agent_rep)

# --------------- End of File ---------------
