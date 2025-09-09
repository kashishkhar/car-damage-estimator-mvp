# Car Damage Estimator — MVP

A minimal full-stack prototype for AI-assisted claims triage. Upload a vehicle photo or paste a URL; the app returns:

 - Vehicle metadata (make, model, color + confidence)

 - Damage summary (human-readable)

 - Estimated cost band with a transparent, itemized breakdown

 - Routing decision (AUTO-APPROVE / INVESTIGATE / SPECIALIST) with a “why” explainer

Demo

 - Deployed (Vercel): [Link to Demo](https://car-damage-estimator-mvp.vercel.app/)

 - Sample images: available in the left panel for one-click testing



## Getting Started
1) Requirements:
- Node 18+ and npm/yarn
- OpenAI API key with GPT-4 Vision access
- (Optional) Roboflow API key + model/version for YOLO boxes

2) Install

```bash
pnpm install
# or
npm i / yarn
```

3) Env Vars (.env.local)

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Models
MODEL_VISION=gpt-4o-mini
MODEL_VEHICLE=gpt-4o-mini
DUAL_PASS_ANALYZE=1
MODEL_VISION_FALLBACK=gpt-4o

# Optional Roboflow YOLO Model
ROBOFLOW_API_KEY=...
ROBOFLOW_MODEL=car-damage-...
ROBOFLOW_VERSION=1
ROBOFLOW_CONF=0.10
ROBOFLOW_OVERLAP=0.30

# Cost & policy knobs (demo defaults)
LABOR_RATE=125
LABOR_CORRECTION=1.4
PAINT_MAT_COST=300
PARTS_BASE=500
PARTS_DYNAMIC=1

AUTO_MAX_COST=1500
AUTO_MAX_SEVERITY=2
AUTO_MIN_CONF=0.75
SPECIALIST_MIN_COST=5000
SPECIALIST_MIN_SEVERITY=4

BLEND_DISCOUNT=0.6
CONTINGENCY_BASE=0.10
CONTINGENCY_FRONT_HEAVY=0.10
CONTINGENCY_REAR_HEAVY=0.05
CONTINGENCY_MAX=0.30

# Client-side bands (for legends)
NEXT_PUBLIC_CONF_HIGH=0.85
NEXT_PUBLIC_CONF_MED=0.60
```

4) Run locally

```bash
pnpm dev
# open http://localhost:3000
```

5) Deploy (Vercel)
- vercel → add env vars in Vercel dashboard → vercel --prod
- If you change env vars later, redeploy to pick them up.

## Architecture Overview
<img width="441" height="587" alt="Screenshot 2025-09-09 at 7 35 03 AM" src="https://github.com/user-attachments/assets/9e5d964e-d7b2-415a-98e8-708a5ddf6b90" />


1) Front-end (Next.js / React).
- Single page UI; image upload/URL; results on the same page
- YOLO overlay (letterbox/object-contain aware)
- Cost breakdown expander (labor, paint/materials, parts detail, contingency)
- Print-friendly “Export Report PDF” (client-side snapshot with overlay)

2) API Routes.
- POST ```/api/detect```
  - Inputs: file or imageUrl
  - Process: optional Roboflow YOLO detection + fast LLM gate (vehicle/quality check)
  - Output: YOLO boxes, quick vehicle guess, issues list (debug information optional)
- POST ```/api/analyze```
  - Inputs: file or imageUrl (+ optional YOLO seeds from /detect)
  - Process: Vision LLM → structured JSON (vehicle + damage_items) → server normalization, dynamic parts pricing, cost breakdown, routing decision
  - Output: Vehicle metadata, damage items, narrative, cost estimate (with breakdown/assumptions), routing decision, summary

- Labor: Σ(est_labor_hours) × LABOR_RATE × LABOR_CORRECTION
- Paint/Materials: PAINT_MAT_COST per affected panel; light blends × BLEND_DISCOUNT
- Parts: Dynamic LLM pricing with conservative sanity-bands by part family; falls back to PARTS_BASE
- Contingency: Up to CONTINGENCY_MAX based on severity/zone (front/rear heavy)
- Variance band: ±15% base; ±25% if severe or contingency applied

4) Routing.
- AUTO-APPROVE if (max severity ≤ AUTO_MAX_SEVERITY) AND (cost_high ≤ AUTO_MAX_COST) AND (agg confidence ≥ AUTO_MIN_CONF)
- SPECIALIST if (max severity ≥ SPECIALIST_MIN_SEVERITY) OR (cost_high ≥ SPECIALIST_MIN_COST)
- Otherwise INVESTIGATE

5) Error handling.
- Human-readable client messages for URL fetch errors, service hiccups, no vehicle detected
- Server retry with short timeouts; simple in-memory rate limiting

6) Security/Privacy (demo).
- No persistent storage; images are processed in-memory and returned
- Add explicit banners: “Visual-only estimate; subject to teardown”

## Data Flow
1) Image Input – User uploads a photo or provides a URL.
2) Dual Detection – Roboflow YOLO detects regions; GPT-4o-mini classifies vehicle presence/quality.
3) Damage Analysis – Vision LLM (GPT-4o-mini, fallback GPT-4o) processes the image with YOLO hints to produce structured damage JSON.
4) Cost Calculation – Dynamic parts pricing, labor rate estimation, paint/materials, and contingency modeling.
5) Business Logic – Routing decision engine applies thresholds (severity, confidence, cost) to classify as AUTO-APPROVE, INVESTIGATE, or SPECIALIST.
6) Response – Returns a comprehensive JSON payload: vehicle metadata, damage items, narrative summary, cost bands with assumptions, and routing recommendation.

## Reasoning behind tool choice
Frontend: Next.js 13 + Tailwind CSS 
- Next.js: Rapid prototyping with file upload, single-page UI, and serverless APIs in one repo. Instant deploys on Vercel with preview links.
- Tailwind: Fastest way to build a clean, enterprise-style UI without design overhead. Keeps styles consistent and flexible.

Backend: Next.js API Routes 
- Single deployable: No extra infra, no CORS issues, shared env. Core APIs: 
 - /api/detect → optional YOLO (Roboflow) detection + vehicle/quality gate 
 - /api/analyze → Vision LLM → normalized JSON → cost engine + routing 
- Guardrails: Confidence validation, normalization, and conservative defaults to ensure consistent outputs.

AI / ML: OpenAI GPT-4o-mini Vision (+ optional YOLO seeding)
- Vision model: Low-latency, lower-cost model that handles vehicle metadata + damage descriptors in one call. 
- YOLO seeds: Bounding boxes (when available) provide spatial hints for the LLM. 
- Fallbacks: Parts pricing banded by family; routing rules tunable via env variables. 

Images: In-memory for MVP
- Prototype: Images handled in-memory for speed and simplicity; no persistent storage. 
- Scalability roadmap: Vercel Blob (for quick persistence) or Amazon S3 (enterprise-grade durability, auditability).

Deployment: Vercel
- Zero-ops: git push → preview → prod. Fast iteration loop.
- Tight Next.js integration: Built-in image handling, env management, rollbacks. 

Development: TypeScript + ESLint
- Type safety: Prevents schema drift between LLM JSON and UI components. 
- Linting/formatting: Keeps repo review-ready and interview-friendly.


## Potential Improvements
- Dual-Pass Auditor Flow: Add a second “auditor” LLM pass when confidence is low to repair JSON, reduce hallucinations, and align outputs to YOLO hints.
- Adaptive Model Selection: Use smaller models (gpt-4o-mini) for simple cases; escalate to full gpt-4o only when confidence drops.
- Multi-Image Normalization: Support multiple photo angles with geometric alignment; combine detections to improve coverage and reduce blind spots.
- Regional Labor & Parts Rates: Integrate shop-specific rate cards or OEM catalogs for more realistic estimates across geographies.
- Total Loss detection: Accurate detection for total loss on a vehicle.
- Caching Layer: Re-use results for identical images (via SHA-256 hash) to avoid duplicate inference costs.
- Persistent Storage: Migrate from in-memory handling to S3/Blob storage with lifecycle management, IAM roles, and auditability.
- Monitoring & Observability: Metrics on latency, model error rates, pass/fail routing accuracy; alerts for SLA breaches.
- VIN History & Integration: Cross-reference claimed damage against vehicle history.
- Fraud Detection ML: Pattern recognition for staged accidents and inflated claims.
- Predictive Analytics: Forecast claim costs based on vehicle age, mileage, and damage patterns.
- Video Assessment: Real-time damage assessment from mobile video streams.
- Webhook Architecture: Event-driven notifications for downstream systems (CRM, workflow tools).
