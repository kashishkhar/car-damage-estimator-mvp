// app/api/analyze/route.ts
// -------------------------------------------------------------------------------------------------
// Car Damage Estimator — Analyze Route (server)
// --------------------------------------------------------------------------------------
// Responsibilities:
// - Accepts an uploaded image (file) or a public image URL
// - Calls OpenAI Vision to produce structured damage JSON
// - Normalizes/repairs the JSON, computes estimate + routing, and returns payload
// - Adds dynamic parts pricing + detailed cost breakdown for UI explainer
// - Business-guardrails: PDR heuristic for minor dents, parts sanity bands, and stable variance policy
// -------------------------------------------------------------------------------------------------

import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import crypto from "crypto";
import type { AnalyzePayload, DamageItem, Vehicle, Part, PartDetail } from "../../types";

export const runtime = "nodejs";

/* ──────────────────────────────────────────────────────────────────────────
 * Errors (stable JSON shape consumed by UI)
 * ------------------------------------------------------------------------ */
const ERR = {
  NO_IMAGE: "E_NO_IMAGE",
  BAD_URL: "E_BAD_URL",
  MODEL_JSON: "E_MODEL_JSON",
  SERVER: "E_SERVER",
} as const;
type ErrorCode = (typeof ERR)[keyof typeof ERR];

function errJson(message: string, code: ErrorCode, status = 400) {
  return NextResponse.json({ error: message, error_code: code }, { status });
}

/* ──────────────────────────────────────────────────────────────────────────
 * Environment / Policy
 * - Server values are mirrored by NEXT_PUBLIC_* on the client, where appropriate.
 * - Defaults picked to be defensible in a carrier demo, not industry-final.
 * ------------------------------------------------------------------------ */
const MODEL_ID = process.env.MODEL_VISION || "gpt-4o-mini";

const LABOR_RATE = Number(process.env.LABOR_RATE ?? 125);
const PAINT_PER_PANEL = Number(process.env.PAINT_MAT_COST ?? 300);

// Routing thresholds
const AUTO_MAX_COST = Number(process.env.AUTO_MAX_COST ?? 1500);
const AUTO_MAX_SEVERITY = Number(process.env.AUTO_MAX_SEVERITY ?? 2);
const AUTO_MIN_CONF = Number(process.env.AUTO_MIN_CONF ?? 0.75);
const SPEC_MIN_COST = Number(process.env.SPECIALIST_MIN_COST ?? 5000);
const SPEC_MIN_SEVERITY = Number(process.env.SPECIALIST_MIN_SEVERITY ?? 4);

// Estimation policy knobs
const LABOR_CORRECTION = Number(process.env.LABOR_CORRECTION ?? 1.4); // capture R&I/setup/omissions
const BLEND_DISCOUNT = Number(process.env.BLEND_DISCOUNT ?? 0.6);     // lighter refinish for light scratch/chip

// Contingency (hidden damage) policy
const CONT_BASE = Number(process.env.CONTINGENCY_BASE ?? 0.10);
const CONT_FRONT_HEAVY = Number(process.env.CONTINGENCY_FRONT_HEAVY ?? 0.10);
const CONT_REAR_HEAVY  = Number(process.env.CONTINGENCY_REAR_HEAVY ?? 0.05);
const CONT_MAX = Number(process.env.CONTINGENCY_MAX ?? 0.30);

// Parts pricing
const PARTS_BASE = Number(process.env.PARTS_BASE ?? 500);
const PARTS_DYNAMIC = Number(process.env.PARTS_DYNAMIC ?? 0) === 1;

// Variance policy (range band)
const VARIANCE_BASE = Number(process.env.VARIANCE_BASE ?? 0.15);     // ±15% typical
const VARIANCE_SEVERE = Number(process.env.VARIANCE_SEVERE ?? 0.25); // ±25% when severe/contingency present

/* ──────────────────────────────────────────────────────────────────────────
 * OpenAI client (lazy init)
 * ------------------------------------------------------------------------ */
let _openai: OpenAI | null = null;
function getOpenAI(): OpenAI {
  if (_openai) return _openai;
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new Error("OPENAI_API_KEY environment variable is missing or empty");
  _openai = new OpenAI({ apiKey: key });
  return _openai;
}

/* ──────────────────────────────────────────────────────────────────────────
 * Utilities / guards
 * ------------------------------------------------------------------------ */
function sha256(buf: Buffer) { return crypto.createHash("sha256").update(buf).digest("hex"); }
function sha256String(s: string) { return crypto.createHash("sha256").update(s, "utf8").digest("hex"); }
function isRecord(v: unknown): v is Record<string, unknown> { return typeof v === "object" && v !== null; }

type BBoxRel = [number, number, number, number];
type PolyRel = [number, number][];

function isBBoxRel(v: unknown): v is BBoxRel {
  return Array.isArray(v) && v.length === 4 && v.every((n) => typeof n === "number" && n >= 0 && n <= 1);
}
function isPolygonRel(v: unknown): v is PolyRel {
  return Array.isArray(v)
    && v.length >= 3 && v.length <= 12
    && v.every((pt) => Array.isArray(pt) && pt.length === 2
      && typeof pt[0] === "number" && pt[0] >= 0 && pt[0] <= 1
      && typeof pt[1] === "number" && pt[1] >= 0 && pt[1] <= 1);
}

type YoloSeed = { bbox_rel: BBoxRel; confidence: number };
function isYoloSeed(v: unknown): v is YoloSeed {
  return isRecord(v) && typeof v["confidence"] === "number" && isBBoxRel(v["bbox_rel"]);
}

/* ──────────────────────────────────────────────────────────────────────────
 * Bodyshop heuristics
 * - hoursFor(): rough base hours by part, scaled by severity
 * - needsPaintFor(): defaults to paint for sev≥2 except glass/lights
 * - minorDentPDR(): PDR for side panels when severity ≤2 and type=dent
 * ------------------------------------------------------------------------ */
function hoursFor(part: Part, sev: number) {
  const base =
    part === "bumper" ? 1.2 :
    part === "door" ? 1.5 :
    part === "fender" ? 1.2 :
    part === "hood" ? 1.4 :
    part === "quarter-panel" ? 2.0 :
    part === "headlight" || part === "taillight" ? 0.6 :
    part === "grille" ? 0.8 :
    part === "mirror" ? 0.5 :
    part === "windshield" ? 1.2 :
    part === "wheel" ? 0.7 :
    part === "trunk" ? 1.4 :
    1.0;

  const sevMult = sev <= 1 ? 0.5 : sev === 2 ? 0.8 : sev === 3 ? 1.0 : sev === 4 ? 1.4 : 1.8;
  return +(base * sevMult).toFixed(2);
}
function needsPaintFor(type: string, sev: number, part: Part) {
  if (["windshield", "headlight", "taillight", "mirror"].includes(part)) return false;
  if (type.includes("scratch") || type.includes("paint")) return true;
  return sev >= 2;
}
function minorDentPDR(d: DamageItem): { usePDR: boolean; adjHours: number; adjPaint: boolean } {
  // PDR if: minor dent (sev ≤2), side panels, and damage_type = dent
  const sidePanel = ["door", "fender", "quarter-panel"].includes(d.part);
  const isMinorDent = d.damage_type === "dent" && d.severity <= 2;
  if (sidePanel && isMinorDent) {
    const base = typeof d.est_labor_hours === "number" ? d.est_labor_hours : hoursFor(d.part, d.severity);
    const adjHours = Math.max(0.5, +(base * 0.6).toFixed(2)); // ~40% faster when PDR applies
    return { usePDR: true, adjHours, adjPaint: false };
  }
  return { usePDR: false, adjHours: d.est_labor_hours, adjPaint: d.needs_paint };
}

/* ──────────────────────────────────────────────────────────────────────────
 * Dynamic parts pricing (single JSON completion) + sanity bands
 * ------------------------------------------------------------------------ */
const PART_BANDS: Record<string, { min: number; max: number }> = {
  // conservative “OEM-equivalent retail” ranges (USD)
  bumper: { min: 250, max: 1200 },
  fender: { min: 150, max: 600 },
  door: { min: 250, max: 900 },
  hood: { min: 250, max: 900 },
  "quarter-panel": { min: 300, max: 1200 },
  headlight: { min: 120, max: 800 },
  taillight: { min: 80, max: 600 },
  grille: { min: 120, max: 500 },
  mirror: { min: 60, max: 300 },
  windshield: { min: 180, max: 600 },
  wheel: { min: 120, max: 900 },
  trunk: { min: 250, max: 900 },

  // “trim/cheap” items — keep realistic (door handle etc.)
  "door handle": { min: 40, max: 180 },
  handle: { min: 40, max: 180 },
  bracket: { min: 30, max: 200 },
  clip: { min: 2, max: 15 },
  emblem: { min: 10, max: 60 },
};

function guardPartPrice(name: string, raw: number): number {
  const norm = name.trim().toLowerCase();
  const band = PART_BANDS[norm];
  const val = Math.max(1, Math.round(raw || PARTS_BASE));
  if (!band) return Math.max(50, val); // global sanity floor
  return Math.min(band.max, Math.max(band.min, val));
}

async function pricePartsDynamicOnce(
  uniqueParts: string[],
  vehicle: Vehicle,
  client: OpenAI
): Promise<Record<string, number>> {
  if (!PARTS_DYNAMIC || uniqueParts.length === 0) return {};

  const sys = `
Return ONLY compact JSON mapping part names to typical US retail OEM-equivalent part prices (USD, integers).
Keys must exactly match the input 'parts' values (lowercase). If unsure, use ${PARTS_BASE}.
Example: {"bumper": 580, "fender": 320}`.trim();

  const vehicleStr = [vehicle?.make || "unknown", vehicle?.model || "unknown"].filter(Boolean).join(" ");
  const userContent = `vehicle: ${vehicleStr || "unknown"}; parts: ${JSON.stringify(uniqueParts)}`;

  try {
    const resp = await client.chat.completions.create({
      model: MODEL_ID,
      temperature: 0,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: sys },
        { role: "user", content: userContent },
      ],
    });

    const raw = resp.choices?.[0]?.message?.content ?? "{}";
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const out: Record<string, number> = {};

    for (const [k, v] of Object.entries(parsed)) {
      const key = String(k).toLowerCase();
      const guarded = guardPartPrice(key, typeof v === "number" && Number.isFinite(v) ? v : PARTS_BASE);
      out[key] = guarded;
    }
    return out;
  } catch {
    return {};
  }
}

/* ──────────────────────────────────────────────────────────────────────────
 * Confidence aggregation & routing
 * ------------------------------------------------------------------------ */
function aggregateConfidence(items: DamageItem[]) {
  let numr = 0, den = 0;
  for (const d of items) {
    const sev = Number(d.severity ?? 1);
    const conf = Number(d.confidence ?? 0.5);
    const w = 1 + 0.2 * (sev - 1);
    numr += conf * w;
    den += w;
  }
  return den ? numr / den : 0.5;
}

function routeDecision(items: DamageItem[], estimate: { cost_high: number }) {
  const maxSev = items.reduce((m, d) => Math.max(m, Number(d.severity ?? 1)), 0);
  const agg = aggregateConfidence(items);
  const hi = Number(estimate?.cost_high ?? 0);

  if (maxSev <= AUTO_MAX_SEVERITY && hi <= AUTO_MAX_COST && agg >= AUTO_MIN_CONF) {
    return {
      label: "AUTO-APPROVE" as const,
      reasons: [
        `severity ≤ ${AUTO_MAX_SEVERITY}`,
        `cost_high ≤ $${AUTO_MAX_COST}`,
        `agg_conf ≥ ${Math.round(AUTO_MIN_CONF * 100)}%`,
      ],
    };
  }
  if (maxSev >= SPEC_MIN_SEVERITY || hi >= SPEC_MIN_COST) {
    return {
      label: "SPECIALIST" as const,
      reasons: [
        ...(maxSev >= SPEC_MIN_SEVERITY ? [`severity ≥ ${SPEC_MIN_SEVERITY}`] : []),
        ...(hi >= SPEC_MIN_COST ? [`cost_high ≥ $${SPEC_MIN_COST}`] : []),
      ],
    };
  }
  return { label: "INVESTIGATE" as const, reasons: [`agg_conf ${Math.round(agg * 100)}%`, `max_severity ${maxSev}`, `cost_high $${hi}`] };
}

/* ──────────────────────────────────────────────────────────────────────────
 * Estimation roll-up with detailed per-part breakdown
 * ------------------------------------------------------------------------ */
function estimateFromItems(
  items: DamageItem[],
  opts?: { partsPriceMap?: Record<string, number> }
) {
  const priceMap = opts?.partsPriceMap || {};

  let laborHours = 0;            // raw hours from line items (may be PDR-adjusted)
  let labor = 0;                 // $ after correction
  let paint = 0;                 // $ per panel
  let partsTotal = 0;

  type CostLine = {
    zone: string;
    part: Part;
    est_labor_hours: number;   // raw hours per line (pre-correction)
    paint_cost: number;        // dollars per panel (with blend discount when applicable)
    parts_cost: number;
  };
  const lines: CostLine[] = [];

  const partCounts: Record<string, number> = {};

  let hasFrontSevere = false;
  let hasRearSevere = false;
  let hasAnySev3Plus = false;

  const isLightBlend = (d: DamageItem) => {
    const t = String(d.damage_type || "");
    return (t.includes("scratch") || t.includes("paint")) && (Number(d.severity ?? 1) <= 2);
  };

  for (const d0 of items) {
    const sev = Number(d0.severity ?? 1);
    const dmgType = String(d0.damage_type || "");
    const pdr = minorDentPDR(d0);

    // Apply PDR adjustments if eligible
    const estHours = typeof pdr.adjHours === "number" ? pdr.adjHours : hoursFor(d0.part, sev);
    const needsPaint = typeof pdr.adjPaint === "boolean" ? pdr.adjPaint : needsPaintFor(dmgType, sev, d0.part);

    laborHours += estHours;

    // paint per panel line
    let paintCostLine = 0;
    if (needsPaint) {
      const factor = isLightBlend(d0) ? BLEND_DISCOUNT : 1.0;
      paintCostLine = Math.round(PAINT_PER_PANEL * factor);
    }

    // parts per line
    const likelyParts = Array.isArray(d0.likely_parts) ? d0.likely_parts.map(String) : [];
    const candidates =
      likelyParts.length > 0
        ? likelyParts
        : sev >= 4 && d0.part && d0.part !== "unknown"
        ? [String(d0.part)]
        : [];

    let partsCostLine = 0;
    for (const raw of candidates) {
      const name = raw.trim().toLowerCase();
      if (!name || /paint/i.test(name)) continue;
      const unit = priceMap[name] ?? guardPartPrice(name, PARTS_BASE);
      partsCostLine += unit;
      partCounts[name] = (partCounts[name] ?? 0) + 1;
    }

    hasAnySev3Plus = hasAnySev3Plus || sev >= 3;
    const z = String(d0.zone || "unknown");
    if (sev >= 4) {
      if (z.startsWith("front")) hasFrontSevere = true;
      if (z.startsWith("rear"))  hasRearSevere = true;
    }

    partsTotal += partsCostLine;

    lines.push({
      zone: z,
      part: d0.part,
      est_labor_hours: +estHours.toFixed(2),
      paint_cost: paintCostLine,
      parts_cost: partsCostLine,
    });
  }

  const laborPreCorrection = Math.round(laborHours * LABOR_RATE);
  labor = Math.round(laborPreCorrection * LABOR_CORRECTION);

  paint = Math.round(lines.reduce((s, l) => s + (l.paint_cost || 0), 0));

  const parts_detail: PartDetail[] = Object.entries(partCounts)
    .map(([name, qty]) => {
      const unit = priceMap[name] ?? guardPartPrice(name, PARTS_BASE);
      return { name, qty, unit_price: unit, line_total: qty * unit };
    })
    .sort((a, b) => b.line_total - a.line_total);

  const partsFromDetail = parts_detail.reduce((s, p) => s + p.line_total, 0);
  if (partsFromDetail !== partsTotal) partsTotal = partsFromDetail;

  // Contingency (hidden/teardown)
  let contPct = 0;
  if (hasAnySev3Plus) contPct += CONT_BASE;
  if (hasFrontSevere)  contPct += CONT_FRONT_HEAVY;
  else if (hasRearSevere) contPct += CONT_REAR_HEAVY;
  contPct = Math.min(contPct, CONT_MAX);

  const contBase = labor + partsTotal; // contingency on labor+parts
  const contingency = Math.round(contBase * contPct);

  const subtotal = labor + paint + partsTotal + contingency;

  // Variance policy: ±15% base; widen to ±25% if severe or contingency is applied
  const variance =
    (items.some((i) => Number(i.severity ?? 1) >= 4) || contingency > 0)
      ? VARIANCE_SEVERE
      : VARIANCE_BASE;

  return {
    currency: "USD" as const,
    cost_low: Math.round(subtotal * (1 - variance)),
    cost_high: Math.round(subtotal * (1 + variance)),
    assumptions: [
      `Labor rate: $${LABOR_RATE}/hr`,
      `Labor correction: ×${LABOR_CORRECTION.toFixed(2)} (R&I/setup/omissions)`,
      `Paint & materials: $${PAINT_PER_PANEL} per panel (light blends ×${BLEND_DISCOUNT})`,
      `Parts pricing: ${Object.keys(priceMap).length ? "dynamic (vehicle/part informed, sanity-banded)" : "baseline midpoint (sanity-banded)"}`,
      `Hidden damage contingency: ${(contPct * 100).toFixed(0)}% on labor+parts`,
      `Variance band: ±${Math.round(variance * 100)}%`,
      "Visual-only estimate; subject to teardown",
    ],
    breakdown: {
      labor,
      labor_pre_correction: laborPreCorrection,
      labor_correction_factor: LABOR_CORRECTION,
      paint,
      paint_units: lines.filter(l => (l.paint_cost || 0) > 0).length,
      blend_discount: BLEND_DISCOUNT,
      parts: Math.round(partsTotal),
      contingency,
      dynamic_parts_used: Object.keys(priceMap).length > 0,
      lines,
      parts_detail: parts_detail.length ? parts_detail : undefined,
    },
  };
}

/* ──────────────────────────────────────────────────────────────────────────
 * POST /api/analyze
 * ------------------------------------------------------------------------ */
export async function POST(req: NextRequest) {
  console.time("analyze_total");
  try {
    const form = await req.formData();
    const file = form.get("file") as File | null;
    const imageUrl = (form.get("imageUrl") as string | null)?.toString().trim() || null;

    // Optional YOLO seeds from client
    const yoloSeedsRaw = (form.get("yolo") as string | null)?.toString() || "";
    let yoloSeeds: YoloSeed[] = [];
    if (yoloSeedsRaw) {
      try {
        const parsed = JSON.parse(yoloSeedsRaw) as unknown;
        if (Array.isArray(parsed)) yoloSeeds = (parsed as unknown[]).filter(isYoloSeed) as YoloSeed[];
      } catch { /* tolerate malformed seeds */ }
    }

    if (!file && !imageUrl) return errJson("No file or imageUrl provided", ERR.NO_IMAGE, 400);

    // Build OpenAI image source + audit hash
    let imageSourceUrl: string;
    let imageHash: string;

    if (file) {
      const arr = await file.arrayBuffer();
      const buf = Buffer.from(arr);
      imageSourceUrl = `data:${file.type || "image/jpeg"};base64,${buf.toString("base64")}`;
      imageHash = sha256(buf);
    } else {
      try {
        const u = new URL(imageUrl!);
        if (!/^https?:$/.test(u.protocol)) return errJson("imageUrl must be http(s)", ERR.BAD_URL, 400);
      } catch { return errJson("Invalid imageUrl", ERR.BAD_URL, 400); }
      imageSourceUrl = imageUrl!;
      imageHash = sha256String(imageUrl!);
    }

    // Strict JSON schema; align to YOLO seeds where applicable
    const system = `
Return ONLY valid JSON matching EXACTLY this shape (no prose, no markdown):

{
  "vehicle": { "make": string | null, "model": string | null, "color": string | null, "confidence": number },
  "damage_items": Array<{
    "zone": "front"|"front-left"|"left"|"rear-left"|"rear"|"rear-right"|"right"|"front-right"|"roof"|"unknown",
    "part": "bumper"|"fender"|"door"|"hood"|"trunk"|"quarter-panel"|"headlight"|"taillight"|"grille"|"mirror"|"windshield"|"wheel"|"unknown",
    "damage_type": "dent"|"scratch"|"crack"|"paint-chips"|"broken"|"bent"|"missing"|"glass-crack"|"unknown",
    "severity": 1|2|3|4|5,
    "confidence": number,
    "est_labor_hours": number,
    "needs_paint": boolean,
    "likely_parts": string[],
    "bbox_rel"?: [number, number, number, number],
    "polygon_rel"?: Array<[number, number]>
  }>,
  "narrative": string,
  "normalization_notes": string
}

Rules:
- Confidence ∈ [0,1]. Severity 1..5 (1=very minor, 5=severe/structural). Be conservative.
- est_labor_hours realistic per item; likely_parts may be empty.
- Geometry normalized [0..1]. Prefer polygon_rel for irregular scratches; else bbox_rel.
- If YOLO_SEEDS are provided, ALIGN your geometry/labels to those regions when applicable; avoid inventing far-away areas.
- No extra keys. JSON only.
`.trim();

    const yoloContext = yoloSeeds.length ? `YOLO_SEEDS: ${JSON.stringify(yoloSeeds.slice(0, 12))}` : `YOLO_SEEDS: []`;

    const client = getOpenAI();
    const completion = await client.chat.completions.create({
      model: MODEL_ID,
      temperature: 0,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: system },
        {
          role: "user",
          content: [
            { type: "text", text: "Analyze this car image and fill the schema. Be concise and conservative." },
            { type: "text", text: yoloContext },
            { type: "image_url", image_url: { url: imageSourceUrl } },
          ],
        },
      ],
    });

    const raw = completion.choices?.[0]?.message?.content ?? "{}";
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      return errJson("Model returned invalid JSON", ERR.MODEL_JSON, 502);
    }

    // Normalize/repair items
    const itemsIn = isRecord(parsed) && Array.isArray((parsed as Record<string, unknown>)["damage_items"])
      ? ((parsed as Record<string, unknown>)["damage_items"] as unknown[])
      : [];

    const items: DamageItem[] = itemsIn.map((dIn) => {
      const d = isRecord(dIn) ? dIn : {};
      const sevRaw = d["severity"];
      const sev = (typeof sevRaw === "number" && sevRaw >= 1 && sevRaw <= 5 ? sevRaw : 2) as 1 | 2 | 3 | 4 | 5;
      const part = (typeof d["part"] === "string" ? d["part"] : "unknown") as Part;
      const estRaw = d["est_labor_hours"];
      const hrs = typeof estRaw === "number" ? estRaw : hoursFor(part, sev);
      const dmgType = (typeof d["damage_type"] === "string" ? d["damage_type"] : "unknown") as DamageItem["damage_type"];
      const needsRaw = d["needs_paint"];
      const needs = typeof needsRaw === "boolean" ? needsRaw : needsPaintFor(String(dmgType || ""), sev, part);
      const confRaw = d["confidence"];
      const conf = typeof confRaw === "number" ? Math.max(0, Math.min(1, confRaw)) : 0.5;

      const out: DamageItem = {
        zone: (typeof d["zone"] === "string" ? d["zone"] : "unknown") as DamageItem["zone"],
        part,
        damage_type: dmgType,
        severity: sev,
        confidence: conf,
        est_labor_hours: hrs,
        needs_paint: needs,
        likely_parts: Array.isArray(d["likely_parts"]) ? (d["likely_parts"] as unknown[]).map(String) : [],
      };

      const bb = d["bbox_rel"];
      if (isBBoxRel(bb)) out.bbox_rel = [bb[0], bb[1], bb[2], bb[3]];
      const poly = d["polygon_rel"];
      if (isPolygonRel(poly)) out.polygon_rel = (poly as PolyRel).map((pt) => [pt[0], pt[1]]);
      return out;
    });

    // Fallback to YOLO seeds if the model returns zero items
    const itemsFinal: DamageItem[] =
      items.length > 0
        ? items
        : yoloSeeds.map((b) => ({
            zone: "unknown",
            part: "unknown",
            damage_type: "unknown",
            severity: 2,
            confidence: Math.max(0, Math.min(1, b.confidence)),
            est_labor_hours: hoursFor("unknown" as Part, 2),
            needs_paint: false,
            likely_parts: [],
            bbox_rel: b.bbox_rel,
          }));

    // Vehicle metadata (optional)
    const vehicleRec = isRecord(parsed) && isRecord((parsed as Record<string, unknown>)["vehicle"])
      ? ((parsed as Record<string, unknown>)["vehicle"] as Record<string, unknown>)
      : {};
    const vehicle: Vehicle = {
      make: typeof vehicleRec["make"] === "string" ? (vehicleRec["make"] as string) : null,
      model: typeof vehicleRec["model"] === "string" ? (vehicleRec["model"] as string) : null,
      color: typeof vehicleRec["color"] === "string" ? (vehicleRec["color"] as string) : null,
      confidence: typeof vehicleRec["confidence"] === "number" ? (vehicleRec["confidence"] as number) : 0,
    };

    // Dynamic parts pricing map (best-effort + sanity bands)
    const uniqueParts = Array.from(
      new Set(
        itemsFinal
          .flatMap((d) => {
            const listed = Array.isArray(d.likely_parts) ? d.likely_parts.map(String) : [];
            if (listed.length) return listed;
            return d.severity >= 4 && d.part && d.part !== "unknown" ? [String(d.part)] : [];
          })
          .map((s) => s.toLowerCase().trim())
          .filter(Boolean)
      )
    );
    const partsPriceMap = await pricePartsDynamicOnce(uniqueParts, vehicle, getOpenAI());

    // Estimate + routing
    const estimate = estimateFromItems(itemsFinal, { partsPriceMap });
    const decision = routeDecision(itemsFinal, estimate);

    const narrative = isRecord(parsed) && typeof (parsed as Record<string, unknown>)["narrative"] === "string"
      ? ((parsed as Record<string, unknown>)["narrative"] as string)
      : "";
    const normalization_notes =
      isRecord(parsed) && typeof (parsed as Record<string, unknown>)["normalization_notes"] === "string"
        ? ((parsed as Record<string, unknown>)["normalization_notes"] as string)
        : "";

    const damage_summary = (
      itemsFinal.length
        ? itemsFinal.map((d) => `${d.zone} ${d.part} — ${d.damage_type}, sev ${d.severity}`).join("; ")
        : narrative
    ).slice(0, 400);

    const payload: AnalyzePayload = {
      schema_version: "1.6.0",
      model: MODEL_ID,
      runId: crypto.randomUUID(),
      image_sha256: imageHash,
      vehicle,
      damage_items: itemsFinal,
      narrative,
      normalization_notes,
      estimate,
      decision,
      damage_summary,
    };

    console.timeEnd("analyze_total");
    return NextResponse.json(payload);
  } catch (e: unknown) {
    console.timeEnd("analyze_total");
    const msg = e instanceof Error ? e.message : "Server error";
    return errJson(msg, ERR.SERVER, 500);
  }
}
