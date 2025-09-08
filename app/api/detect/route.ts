// app/api/detect/route.ts
// --------------------------------------------------------------------------------------
// Car Damage Estimator — Detect Route (server)
// --------------------------------------------------------------------------------------
// Responsibilities:
// - Accept image file or URL
// - (Optional) Run Roboflow model to get lightweight YOLO-style boxes
// - Run a fast OpenAI classifier: vehicle present? image quality OK? (usable?)
// - Return a compact DetectPayload that feeds the /api/analyze step
// --------------------------------------------------------------------------------------

import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import crypto from "crypto";
import { DetectPayload, Vehicle, YoloBoxRel } from "../../types";
import { ERR, errJson, isRecord } from "../_shared";

export const runtime = "nodejs";

/* ──────────────────────────────────────────────────────────────────────────
 * Config & Environment
 * ------------------------------------------------------------------------ */

const MODEL_VEHICLE = process.env.MODEL_VEHICLE || "gpt-4o-mini"; // small/fast classifier

// Roboflow settings (optional; this route tolerates empty keys)
const ROBOFLOW_API_KEY = process.env.ROBOFLOW_API_KEY || "";
const ROBOFLOW_MODEL = process.env.ROBOFLOW_MODEL || "";
const ROBOFLOW_VERSION = process.env.ROBOFLOW_VERSION || "";
const ROBOFLOW_CONF = process.env.ROBOFLOW_CONF ?? "";
const ROBOFLOW_OVERLAP = process.env.ROBOFLOW_OVERLAP ?? "";

/* ──────────────────────────────────────────────────────────────────────────
 * OpenAI client (lazy)
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
 * Utilities
 * ------------------------------------------------------------------------ */

type UnknownRecord = Record<string, unknown>;

function getNum(obj: Record<string, unknown>, key: string): number | undefined {
  return typeof obj[key] === "number" ? (obj[key] as number) : undefined;
}

function sha256(buf: Buffer) { return crypto.createHash("sha256").update(buf).digest("hex"); }
function sha256String(s: string) { return crypto.createHash("sha256").update(s, "utf8").digest("hex"); }

/** Best-effort extraction across Roboflow response variants. */
function extractPredictions(obj: unknown): unknown[] {
  if (!isRecord(obj)) return [];
  if (Array.isArray((obj as UnknownRecord).predictions)) return (obj as UnknownRecord).predictions as unknown[];
  if (
    isRecord((obj as UnknownRecord).result) &&
    Array.isArray(((obj as UnknownRecord).result as UnknownRecord).predictions)
  ) {
    return ((obj as UnknownRecord).result as UnknownRecord).predictions as unknown[];
  }
  if (Array.isArray((obj as UnknownRecord).outputs)) return (obj as UnknownRecord).outputs as unknown[];
  return [];
}

/* ──────────────────────────────────────────────────────────────────────────
 * Roboflow: Hosted inference → normalized YOLO boxes (with debug)
 * ------------------------------------------------------------------------ */

type RoboflowDebug = {
  enabled: boolean;
  missing_env?: string[];
  rf_url?: string;
  status?: number;
  ok?: boolean;
  body_snippet?: string;
  parse_path?: "predictions" | "result.predictions" | "outputs" | "none";
  parsed_count?: number;
  error?: string;
  params?: { confidence: string; overlap: string };
  sent_mode?: "base64_body" | "image_query";
  image_dims?: { width: number | null; height: number | null };
};

async function detectWithRoboflowDebug(imageUrlOrDataUrl: string): Promise<{ boxes: YoloBoxRel[]; debug: RoboflowDebug; }> {
  const debug: RoboflowDebug = { enabled: true };
  const boxes: YoloBoxRel[] = [];

  // ENV guard
  const missing: string[] = [];
  if (!ROBOFLOW_API_KEY) missing.push("ROBOFLOW_API_KEY");
  if (!ROBOFLOW_MODEL) missing.push("ROBOFLOW_MODEL");
  if (!ROBOFLOW_VERSION) missing.push("ROBOFLOW_VERSION");
  if (missing.length) {
    debug.missing_env = missing;
    return { boxes, debug };
  }

  try {
    // Build the base URL and apply optional tuning
    let baseUrl = `https://detect.roboflow.com/${ROBOFLOW_MODEL}/${ROBOFLOW_VERSION}?api_key=${ROBOFLOW_API_KEY}`;

    if (ROBOFLOW_CONF)    baseUrl += `&confidence=${encodeURIComponent(ROBOFLOW_CONF)}`;
    if (ROBOFLOW_OVERLAP) baseUrl += `&overlap=${encodeURIComponent(ROBOFLOW_OVERLAP)}`;

    // record params for client debug panel
    debug.params = {
      confidence: ROBOFLOW_CONF || "(default)",
      overlap: ROBOFLOW_OVERLAP || "(default)",
    };

    // Decide request shape
    let url = baseUrl;
    let body: string | undefined;
    const headers: Record<string, string> = {};
    let sent_mode: RoboflowDebug["sent_mode"] = undefined;

    if (imageUrlOrDataUrl.startsWith("data:")) {
      // Base64 uploads → body must be raw base64 (strip data: header)
      const comma = imageUrlOrDataUrl.indexOf(",");
      const rawB64 = comma >= 0 ? imageUrlOrDataUrl.slice(comma + 1) : imageUrlOrDataUrl;
      body = rawB64;
      headers["Content-Type"] = "application/x-www-form-urlencoded";
      sent_mode = "base64_body";
    } else {
      // Hosted image URL → goes on the query string
      url = `${baseUrl}&image=${encodeURIComponent(imageUrlOrDataUrl)}`;
      sent_mode = "image_query";
    }

    debug.rf_url = url;
    debug.sent_mode = sent_mode;

    const res = await fetch(url, { method: "POST", headers, body });

    debug.status = res.status;
    debug.ok = res.ok;

    const text = await res.text();
    debug.body_snippet = text.slice(0, 240);

    if (!res.ok) return { boxes, debug };

    // Parse, extract predictions, normalize to [0..1] boxes
    let parsed: unknown;
    try { parsed = JSON.parse(text); } catch {
      debug.error = "json_parse_failed";
      return { boxes, debug };
    }

    // Grab global image dimensions if present: { image: { width, height } }
    const imgObj = isRecord(parsed) && isRecord((parsed as UnknownRecord).image)
      ? ((parsed as UnknownRecord).image as UnknownRecord)
      : {};
    const globalW = getNum(imgObj, "width");
    const globalH = getNum(imgObj, "height");
    debug.image_dims = { width: globalW ?? null, height: globalH ?? null };

    let path: RoboflowDebug["parse_path"] = "none";
    const raw = extractPredictions(parsed);
    if (Array.isArray(raw)) {
      if (Array.isArray((parsed as UnknownRecord).predictions)) path = "predictions";
      else if (
        isRecord((parsed as UnknownRecord).result) &&
        Array.isArray(((parsed as UnknownRecord).result as UnknownRecord).predictions)
      ) path = "result.predictions";
      else if (Array.isArray((parsed as UnknownRecord).outputs)) path = "outputs";
    }
    debug.parse_path = path;

    const preds = raw.filter(isRecord) as Record<string, unknown>[];
    debug.parsed_count = preds.length;

    for (const p of preds) {
      const conf = getNum(p, "confidence") ?? getNum(p, "conf") ?? 0.5;

      // Preferred: center (x,y) + size (width,height) in pixels, normalize by image dims.
      // Use per-pred image_width/image_height if present, else fallback to top-level image.width/height.
      const pxW = getNum(p, "image_width") ?? globalW;
      const pxH = getNum(p, "image_height") ?? globalH;

      if (
        getNum(p, "x") !== undefined && getNum(p, "y") !== undefined &&
        getNum(p, "width") !== undefined && getNum(p, "height") !== undefined &&
        pxW !== undefined && pxH !== undefined && pxW > 0 && pxH > 0
      ) {
        const cx = getNum(p, "x") as number;
        const cy = getNum(p, "y") as number;
        const w  = getNum(p, "width") as number;
        const h  = getNum(p, "height") as number;
        const W  = pxW as number;
        const H  = pxH as number;

        const nx = Math.max(0, Math.min(1, (cx - w / 2) / W));
        const ny = Math.max(0, Math.min(1, (cy - h / 2) / H));
        const nw = Math.max(0, Math.min(1, w / W));
        const nh = Math.max(0, Math.min(1, h / H));

        boxes.push({ bbox_rel: [nx, ny, nw, nh], confidence: conf! });
        continue;
      }

      // Fallback min/max normalized
      if (
        getNum(p, "x_min") !== undefined && getNum(p, "x_max") !== undefined &&
        getNum(p, "y_min") !== undefined && getNum(p, "y_max") !== undefined
      ) {
        const xmin = getNum(p, "x_min") as number;
        const ymin = getNum(p, "y_min") as number;
        const xmax = getNum(p, "x_max") as number;
        const ymax = getNum(p, "y_max") as number;

        const nx = Math.max(0, Math.min(1, xmin));
        const ny = Math.max(0, Math.min(1, ymin));
        const nw = Math.max(0, Math.min(1, xmax - xmin));
        const nh = Math.max(0, Math.min(1, ymax - ymin));
        boxes.push({ bbox_rel: [nx, ny, nw, nh], confidence: conf! });
      }
    }

    return { boxes, debug };
  } catch (e: unknown) {
    debug.error = e instanceof Error ? e.message : "roboflow_call_failed";
    return { boxes, debug };
  }
}

/* ──────────────────────────────────────────────────────────────────────────
 * OpenAI quick gate: Vehicle present? Quality OK?
 * ------------------------------------------------------------------------ */

async function quickQualityGate(imageUrlOrDataUrl: string): Promise<{
  is_vehicle: boolean;
  quality_ok: boolean;
  issues: string[];
  vehicle_guess: Vehicle;
}> {
  const system = `
Return ONLY JSON with this shape:

{
  "is_vehicle": boolean,
  "quality_ok": boolean,
  "issues": string[],
  "vehicle": { "make": string|null, "model": string|null, "color": string|null, "confidence": number }
}

Rules:
- "issues" can include: "not_vehicle", "blurry", "low_light", "heavy_occlusion", "cropped", "too_small".
- If unsure about make/model/color, set null but always provide numeric "confidence" [0..1].
- Be conservative; JSON only.
`.trim();

  const client = getOpenAI();
  const completion = await client.chat.completions.create({
    model: MODEL_VEHICLE,
    temperature: 0,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: system },
      {
        role: "user",
        content: [
          { type: "text", text: "Classify whether this is a usable car image for damage assessment." },
          { type: "image_url", image_url: { url: imageUrlOrDataUrl } },
        ],
      },
    ],
  });

  let parsed: unknown = {};
  try {
    parsed = JSON.parse(completion.choices?.[0]?.message?.content ?? "{}");
  } catch {
    parsed = {
      is_vehicle: true,
      quality_ok: true,
      issues: [],
      vehicle: { make: null, model: null, color: null, confidence: 0.6 },
    };
  }

  const pv = (typeof parsed === "object" && parsed !== null ? parsed : {}) as UnknownRecord;
  const vehicleObj = (pv["vehicle"] && typeof pv["vehicle"] === "object" ? (pv["vehicle"] as UnknownRecord) : {});

  const vehicle_guess: Vehicle = {
    make: typeof vehicleObj["make"] === "string" ? (vehicleObj["make"] as string) : null,
    model: typeof vehicleObj["model"] === "string" ? (vehicleObj["model"] as string) : null,
    color: typeof vehicleObj["color"] === "string" ? (vehicleObj["color"] as string) : null,
    confidence: typeof vehicleObj["confidence"] === "number" ? (vehicleObj["confidence"] as number) : 0.6,
  };

  return {
    is_vehicle: typeof pv["is_vehicle"] === "boolean" ? (pv["is_vehicle"] as boolean) : true,
    quality_ok: typeof pv["quality_ok"] === "boolean" ? (pv["quality_ok"] as boolean) : true,
    issues: Array.isArray(pv["issues"]) ? (pv["issues"] as unknown[]).map(String) : [],
    vehicle_guess,
  };
}

/* ──────────────────────────────────────────────────────────────────────────
 * POST /api/detect
 * ------------------------------------------------------------------------ */

export async function POST(req: NextRequest) {
  console.time("detect_total");
  try {
    const form = await req.formData();
    const file = form.get("file") as File | null;
    const imageUrl = (form.get("imageUrl") as string | null)?.toString().trim() || null;

    if (!file && !imageUrl) {
      return errJson("No file or imageUrl provided", ERR.NO_IMAGE, 400);
    }

    // Build a safe source URL and an audit hash (consistent with /api/analyze).
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
      } catch {
        return errJson("Invalid imageUrl", ERR.BAD_URL, 400);
      }
      imageSourceUrl = imageUrl!;
      imageHash = sha256String(imageUrl!);
    }

    // Run Roboflow detection + OpenAI quick gate in parallel.
    const [rf, q] = await Promise.all([
      detectWithRoboflowDebug(imageSourceUrl),
      quickQualityGate(imageSourceUrl),
    ]);

    const yolo_boxes = rf.boxes;

    // Merge quick gate issues with detect status
    const issues = q.issues.slice();
    if (yolo_boxes.length === 0) {
      issues.push("no_yolo_detections");
    }

    // Build response payload (include debug for client UI)
    const payload: DetectPayload & { yolo_debug?: RoboflowDebug } = {
      model: MODEL_VEHICLE,
      runId: crypto.randomUUID(),
      image_sha256: imageHash,
      yolo_boxes,
      vehicle: q.vehicle_guess,
      is_vehicle: q.is_vehicle,
      has_damage: yolo_boxes.length > 0,
      quality_ok: q.quality_ok,
      issues,
      yolo_debug: rf.debug,
    };

    console.timeEnd("detect_total");
    return NextResponse.json(payload);
  } catch (e: unknown) {
    console.timeEnd("detect_total");
    const message = e instanceof Error ? e.message : "detect error";
    return errJson(message, ERR.SERVER, 500);
  }
}
