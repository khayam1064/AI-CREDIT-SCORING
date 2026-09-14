with open("SYSTEM_ARCHITECTURE_AND_MODELS_GUIDE.md", "r", encoding="utf-8") as f:
    guide = f.read()

# Append Production API & Monitoring section to the guide
api_docs = """

---

## 7. Commercial Production Microservice & Real-Time REST API

In Tier-1 banking deployments, the scoring models are served as a standalone, containerized **FastAPI REST microservice** (Port `8080`), decoupled from the web presentation UI.

### Production Endpoints:
* `GET /health`: Microservice readiness, engine version (`3.2.0-Production`), standard compliance checks.
* `GET /api/v1/customers/{customer_id}`: Real-time full decisioning dossier (Composite score, risk grade, calibrated PD, P10/P50/P90 income quantiles, multi-bound limits, APR breakdown, and SHAP adverse action codes).
* `POST /api/v1/score/underwrite`: Headless loan origination endpoint for mobile banking apps & core banking integrations (Temenos, Mambu, Thought Machine).
* `GET /api/v1/portfolio/ifrs9`: Account-level dynamic EAD impairment calculations with multi-scenario macroeconomic stress testing (Baseline, Downturn, Upturn).
* `GET /api/v1/portfolio/drift-monitoring`: Automated Population Stability Index (PSI) and Characteristic Feature Drift tracking.

### Interactive OpenAPI Documentation:
* Swagger UI: `http://localhost:8080/docs`
* ReDoc Specification: `http://localhost:8080/redoc`
"""

if "Commercial Production Microservice" not in guide:
    guide += api_docs
    with open("SYSTEM_ARCHITECTURE_AND_MODELS_GUIDE.md", "w", encoding="utf-8") as f:
        f.write(guide)
    print("Updated SYSTEM_ARCHITECTURE_AND_MODELS_GUIDE.md with REST API & Monitoring documentation!")
else:
    print("Documentation already up to date.")
