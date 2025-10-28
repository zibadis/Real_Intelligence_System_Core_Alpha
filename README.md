
# Real Intelligence System — Core Alpha

**RIS-Core-Alpha** is the first open, no-login, adaptive core for Real Intelligence Systems.  
Built entirely on **FastAPI**, it unifies multiple domains (Security, Economy, Health) through a shared **DHAL (Domain Homogenization & Adaptation Layer)**.

---

## 🌐 Overview
RIS-Core-Alpha processes real-time or static data pulses through a spherical data grid ("beads"),  
analyzes saliency patterns, issues alerts, and evolves adaptively through control parameters:
`τ (threshold), κ (consensus), Γ (throttle), λ (decay)`.

It operates in fully open mode — **no login, no external API key**, ready for local or global scaling.

---

## 🧠 Core Features
- 🧩 **DHAL → CFV (I,V,C,A,E)** unification across all domains  
- ⚙️ **Adaptive parameters** based on Precision/Recall/FIR feedback  
- 💬 **FastAPI endpoints:** `/pulse`, `/alerts`, `/recommendations`, `/snapshot`, `/telemetry`  
- 📊 **Self-healing grid** with shadow→main promotion and auto decay  
- 🔒 No user data, no authentication — purely logic-based intelligence

---

## 🧰 Installation

```bash
git clone https://github.com/<your-username>/Real_Intelligence_System_Core_Alpha.git
cd Real_Intelligence_System_Core_Alpha
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
