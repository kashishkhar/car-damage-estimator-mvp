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
// - MODEL / VERSION select the hosted inference endpoint
// - CONF / OVERLAP are pass-through thresholds (safe to omit → provider defaults)
const ROBOFLOW_API_KEY = process.env.ROBOFLOW_API_KEY || "";
const ROBOFLOW_MODEL = process.env.ROBOFLOW_MODEL || "";
const ROBOFLOW_VERSION = process.env.ROBOFLOW_VERSION || "";
const ROBOFLOW_CONF = process.env.ROBOFLOW_CONF ?? "";       // e.g., "0.10"
const ROBOFLOW_OVERLAP = process.env.ROBOFLOW_OVERLAP ?? ""; // e.g., "0.30"

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
 * Utilities
 * ------------------------------------------------------------------------ */

function getNum(obj: Record<string, unknown>, key: string): number | undefined {
  return typeof obj[key] === "number" ? (obj[key] as number) : undefined;
}

function sha256(buf: Buffer) { return crypto.createHash("sha256").update(buf).digest("hex"); }
function sha256String(s: string) { return crypto.createHash("sha256").update(s, "utf8").digest("hex"); }

/** Best-effort extraction across Roboflow response variants (predictions array may live in different roots). */
function extractPredictions(obj: unknown): unknown[] {
  if (!isRecord(obj)) return [];
  if (Array.isArray(obj.predictions)) return obj.predictions as unknown[];
  if (isRecord(obj.result) && Array.isArray(obj.result.predictions)) return obj.result.predictions as unknown[];
  if (Array.isArray(obj.outputs)) return obj.outputs as unknown[];
  return [];
}

/* ──────────────────────────────────────────────────────────────────────────
 * Roboflow: Hosted inference → normalized YOLO boxes
 * ------------------------------------------------------------------------
 * Behavior:
 * - Uploads: POST raw base64 in body (strip "data:*;base64," prefix). No ?image= key.
 * - URLs:    POST with ?image=<http(s)://...> on the query string; empty body.
 * - Normalization: predictions may return pixel-space (x,y,width,height) with
 *   image dims only at the top-level ({ image: { width, height }}). We normalize
 *   to [0..1] using per-pred dims if present, else the global image dims.
 * - Tuning: optional confidence/overlap envs are appended to the request URL.
 * ------------------------------------------------------------------------ */

async function detectWithRoboflow(imageUrlOrDataUrl: string): Promise<YoloBoxRel[]> {
  // Tolerate missing keys → feature is optional
  if (!ROBOFLOW_API_KEY || !ROBOFLOW_MODEL || !ROBOFLOW_VERSION) return [];

  // Build base URL with API key + optional threshold knobs
  let baseUrl = `https://detect.roboflow.com/${ROBOFLOW_MODEL}/${ROBOFLOW_VERSION}?api_key=${ROBOFLOW_API_KEY}`;
  if (ROBOFLOW_CONF)    baseUrl += `&confidence=${encodeURIComponent(ROBOFLOW_CONF)}`;
  if (ROBOFLOW_OVERLAP) baseUrl += `&overlap=${encodeURIComponent(ROBOFLOW_OVERLAP)}`;

  // Decide request shape
  let url = baseUrl;
  let body: string | undefined;
  let headers: Record<string, string> = {};

  if (imageUrlOrDataUrl.startsWith("data:")) {
    // Base64 upload: body = raw b64 (strip "data:*;base64,")
    const comma = imageUrlOrDataUrl.indexOf(",");
    const rawB64 = comma >= 0 ? imageUrlOrDataUrl.slice(comma + 1) : imageUrlOrDataUrl;
    body = rawB64;
    headers["Content-Type"] = "application/x-www-form-urlencoded";
  } else {
    // Hosted URL: pass via query string (?image=...)
    url = `${baseUrl}&image=${encodeURIComponent(imageUrlOrDataUrl)}`;
  }

  try {
    const res = await fetch(url, { method: "POST", headers, body });
    if (!res.ok) return []; // soft-fail → no boxes
    const text = await res.text();

    // Parse JSON (tolerate parse failure → no boxes)
    let parsed: unknown;
    try { parsed = JSON.parse(text); } catch { return []; }

    // Top-level image dims fallback: { image: { width, height } }
    const imgRec = isRecord(parsed) && isRecord((parsed as any).image)
      ? ((parsed as any).image as Record<string, unknown>)
      : {};
    const globalW = getNum(imgRec, "width");
    const globalH = getNum(imgRec, "height");

    // Extract predictions across possible shapes
    const rawPreds = extractPredictions(parsed);
    const preds = rawPreds.filter(isRecord) as Record<string, unknown>[];

    const boxes: YoloBoxRel[] = [];
    for (const p of preds) {
      const conf = getNum(p, "confidence") ?? getNum(p, "conf") ?? 0.5;

      // Preferred center format (pixels) → normalize using per-pred dims OR global dims
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

        // Convert center-format pixels → normalized [x,y,w,h] origin at top-left
        const nx = Math.max(0, Math.min(1, (cx - w / 2) / W));
        const ny = Math.max(0, Math.min(1, (cy - h / 2) / H));
        const nw = Math.max(0, Math.min(1, w / W));
        const nh = Math.max(0, Math.min(1, h / H));
        boxes.push({ bbox_rel: [nx, ny, nw, nh], confidence: conf! });
        continue;
      }

      // Fallback: min/max corners (already normalized in [0..1] space)
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

    return boxes;
  } catch {
    // Network/timeout/etc → treat as "no detections"
    return [];
  }
}

/* ──────────────────────────────────────────────────────────────────────────
 * OpenAI quick gate: Vehicle present? Quality OK?
 * ------------------------------------------------------------------------
 * Returns a conservative JSON gate to avoid analyzing unusable images.
 * - "issues" may include blur/occlusion/low_light/etc. (best-effort)
 * - "vehicle_guess" is a lightweight hint; UI uses it as metadata.
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

  // Parse model JSON (fallbacks are deliberately permissive)
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

  const pv = (typeof parsed === "object" && parsed !== null ? parsed : {}) as Record<string, unknown>;
  const vehicleObj = (pv["vehicle"] && typeof pv["vehicle"] === "object" ? (pv["vehicle"] as Record<string, unknown>) : {});

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
 * ------------------------------------------------------------------------
 * Input:
 * - multipart/form-data containing either:
 *   • file: File (image/*)
 *   • imageUrl: string (http/https)
 *
 * Output (DetectPayload):
 * - yolo_boxes: normalized YOLO boxes from Roboflow (may be empty)
 * - vehicle / is_vehicle / quality_ok: quick gate hints
 * - issues: best-effort quality issues (array of strings)
 * - has_damage: inferred from yolo_boxes.length > 0 (UX hint)
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

    // Build a safe source URL (data: for uploads; http(s) passthrough for links) + audit hash
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

    // Run Roboflow detection + OpenAI quick gate in parallel
    const [yolo_boxes, q] = await Promise.all([
      detectWithRoboflow(imageSourceUrl),
      quickQualityGate(imageSourceUrl),
    ]);

    const payload: DetectPayload = {
      model: MODEL_VEHICLE,
      runId: crypto.randomUUID(),
      image_sha256: imageHash,
      yolo_boxes,
      vehicle: q.vehicle_guess,
      is_vehicle: q.is_vehicle,
      has_damage: yolo_boxes.length > 0,
      quality_ok: q.quality_ok,
      issues: q.issues,
    };

    console.timeEnd("detect_total");
    return NextResponse.json(payload);
  } catch (e: unknown) {
    console.timeEnd("detect_total");
    const message = e instanceof Error ? e.message : "detect error";
    return errJson(message, ERR.SERVER, 500);
  }
}
