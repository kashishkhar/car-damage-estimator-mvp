// app/types.ts
// -------------------------------------------------------------------------------------------------
// Shared type definitions for the Car Damage Estimator.
// These types are consumed by both server routes (/api/detect, /api/analyze) and the client UI.
// Keep this file as the single source of truth to avoid drift.
// -------------------------------------------------------------------------------------------------

/* ──────────────────────────────────────────────────────────────────────────
 * Common primitives
 * ------------------------------------------------------------------------ */

/** Normalized YOLO bounding box (x, y, w, h) in [0..1] space. */
export type YoloBoxRel = {
  bbox_rel: [number, number, number, number];
  confidence: number; // 0..1
};

export type Vehicle = {
  make: string | null;
  model: string | null;
  color: string | null;
  confidence: number; // 0..1
};

export type ApiError = { error: string; error_code?: string };

/* ──────────────────────────────────────────────────────────────────────────
 * Detect route: optional Roboflow debug block (for client debug panel)
 * ------------------------------------------------------------------------ */

export type RoboflowDebug = {
  enabled: boolean;
  /** Missing required env vars on the server (if any). */
  missing_env?: string[];
  /** Fully-resolved Roboflow request URL (with query params). */
  rf_url?: string;
  /** HTTP status returned by Roboflow. */
  status?: number;
  /** Convenience mirror of status in boolean form. */
  ok?: boolean;
  /** First 240 chars of the raw response body for quick inspection. */
  body_snippet?: string;
  /** Where predictions were parsed from in the response object. */
  parse_path?: "predictions" | "result.predictions" | "outputs" | "none";
  /** Number of predictions parsed (post-filter). */
  parsed_count?: number;
  /** Any parse/network error string captured by the server. */
  error?: string;
  /** Parameters actually sent (confidence/overlap); "(default)" if omitted. */
  params?: { confidence: string; overlap: string };
  /** How the image was sent to Roboflow. */
  sent_mode?: "base64_body" | "image_query";
  /** Top-level image dimensions if provided by Roboflow. */
  image_dims?: { width: number | null; height: number | null };
};

/* ──────────────────────────────────────────────────────────────────────────
 * Detect route payload
 * ------------------------------------------------------------------------ */

export type DetectPayload = {
  model: string;        // classifier model id
  runId: string;
  image_sha256: string;
  yolo_boxes: YoloBoxRel[];
  vehicle: Vehicle;     // quick guess (for UX hints)
  is_vehicle: boolean;  // gate: is the image a vehicle?
  has_damage: boolean;  // inferred: yolo_boxes.length > 0
  quality_ok: boolean;  // usable image?
  issues: string[];     // e.g., ["blurry","low_light","no_yolo_detections"]
  /** Optional debug info from Roboflow (present when enabled server-side). */
  yolo_debug?: RoboflowDebug;
};

/* ──────────────────────────────────────────────────────────────────────────
 * Analyze route primitives
 * ------------------------------------------------------------------------ */

export type Zone =
  | "front" | "front-left" | "left" | "rear-left" | "rear"
  | "rear-right" | "right" | "front-right" | "roof" | "unknown";

export type Part =
  | "bumper" | "fender" | "door" | "hood" | "trunk"
  | "quarter-panel" | "headlight" | "taillight" | "grille"
  | "mirror" | "windshield" | "wheel" | "unknown";

export type DamageType =
  | "dent" | "scratch" | "crack" | "paint-chips"
  | "broken" | "bent" | "missing" | "glass-crack" | "unknown";

/** A single modeled damage item returned by the vision model. */
export type DamageItem = {
  zone: Zone;
  part: Part;
  damage_type: DamageType;
  severity: 1 | 2 | 3 | 4 | 5;
  confidence: number; // 0..1
  est_labor_hours: number;
  needs_paint: boolean;
  likely_parts: string[];
  bbox_rel?: [number, number, number, number];
  polygon_rel?: [number, number][];
};

/* ──────────────────────────────────────────────────────────────────────────
 * Cost model & breakdown (server → client, used by Cost Breakdown UI)
 * ------------------------------------------------------------------------ */

export type EstimateBreakdownLine = {
  /** Zone label (used by UI; paint may be deduped per panel on server). */
  zone: string;
  /** Primary panel/part the line refers to (for context only). */
  part: Part;
  /** Hours used for labor $ calculation (pre-correction). */
  est_labor_hours: number;
  /** Paint $ applied to this line (UI totals leverage paint_units). */
  paint_cost: number;
  /** Parts $ for this line (sum of candidate parts for this damage). */
  parts_cost: number;
};

export type PartDetail = {
  name: string;       // normalized (lowercase)
  qty: number;        // occurrences across damage items
  unit_price: number; // price used per unit (dynamic or baseline, sanity-banded)
  line_total: number; // qty * unit_price
};

export type EstimateBreakdown = {
  labor: number;                     // corrected labor $
  labor_pre_correction: number;      // baseHours * laborRate
  labor_correction_factor: number;   // e.g., 1.40
  paint: number;                     // total paint & materials $
  paint_units: number;               // server-deduped panel count
  blend_discount: number;            // e.g., 0.6 for light blends
  parts: number;                     // total parts $
  contingency: number;               // hidden/teardown $
  dynamic_parts_used: boolean;       // true if dynamic pricing was attempted
  lines: EstimateBreakdownLine[];    // per-damage line summary
  parts_detail?: PartDetail[];       // optional per-part detail (when dynamic used)
};

export type Estimate = {
  currency: "USD";
  cost_low: number;
  cost_high: number;
  /** Human-readable assumptions the UI shows inline. */
  assumptions: string[];
  /** Structured numbers to render the Cost Breakdown expander. */
  breakdown: EstimateBreakdown;
};

/* ──────────────────────────────────────────────────────────────────────────
 * Final analyze payload
 * ------------------------------------------------------------------------ */

export type Decision =
  | { label: "AUTO-APPROVE"; reasons: string[] }
  | { label: "INVESTIGATE"; reasons: string[] }
  | { label: "SPECIALIST"; reasons: string[] };

export type AnalyzePayload = {
  schema_version: string;
  model: string;
  runId: string;
  image_sha256: string;
  vehicle: Vehicle;
  damage_items: DamageItem[];
  narrative: string;
  normalization_notes: string;
  estimate: Estimate;
  decision: Decision;
  /** 1–2 line summary for copy/paste; ~400 chars max. */
  damage_summary: string;
};
