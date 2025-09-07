/**
 * Shared type definitions for the Car Damage Estimator.
 * These types are imported by both the server routes and client UI.
 */

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
  issues: string[];     // e.g., ["blurry","low_light"]
};

/* ──────────────────────────────────────────────────────────────────────────
 * Analyze route payload primitives
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

/** A single modeled damage item returned by the vision LLM. */
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
 * Cost model & breakdown (used by Cost Breakdown UI)
 * ------------------------------------------------------------------------ */

export type CostBreakdownLine = {
  /** Zone label used for de-duping paint/materials per zone. */
  zone: string;
  /** Primary panel/part the line refers to (for context). */
  part: Part;
  /** Hours used for labor $$ calculation. */
  est_labor_hours: number;
  /** Paint $ applied to this line (totals only count each zone once). */
  paint_cost: number;
  /** Parts $ for this line (sum of all candidate parts for this damage). */
  parts_cost: number;
};

/** Itemized dynamic parts detail (used for UI display). */
export type PartDetail = {
  name: string;       // normalized (lowercase)
  qty: number;        // occurrences across damage items
  unit_price: number; // price used per unit (dynamic or baseline)
  line_total: number; // qty * unit_price
};

export type CostBand = {
  currency: "USD";
  cost_low: number;
  cost_high: number;
  /** Human-readable assumptions that the UI shows inline. */
  assumptions: string[];
  /** Optional: structured numbers to render the Cost Breakdown expander. */
  breakdown?: {
    labor: number;
    paint: number;
    parts: number;
    dynamic_parts_used: boolean;
    lines: CostBreakdownLine[];
    parts_detail?: PartDetail[];
  };
};

/* ──────────────────────────────────────────────────────────────────────────
 * Final analyze payload
 * ------------------------------------------------------------------------ */

export type Decision =
  | { label: "AUTO-APPROVE"; reasons: string[] }
  | { label: "INVESTIGATE"; reasons: string[] }
  | { label: "SPECIALIST"; reasons: string[] };

/** Canonical server response used by the client UI. */
export type AnalyzePayload = {
  schema_version: string;
  model: string;
  runId: string;
  image_sha256: string;
  vehicle: Vehicle;
  damage_items: DamageItem[];
  narrative: string;
  normalization_notes: string;
  estimate: CostBand;
  decision: Decision;
  /** 1–2 line summary for copy/paste; ~400 chars max. */
  damage_summary: string;
};