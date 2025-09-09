/* eslint-disable @next/next/no-img-element */
"use client";

/**
 * Car Damage Estimator — Client UI
 * Flow: Upload/URL → /api/detect → /api/analyze
 * - YOLO overlay (object-contain letterbox aware)
 * - Decision/routing explainer (policy mirrored via NEXT_PUBLIC_*)
 * - Cost band + explainer; derives visible rates from numeric totals
 * - Print-friendly report with overlay snapshot
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  AnalyzePayload, DamageItem, YoloBoxRel, RoboflowDebug,
} from "./types";

/* ──────────────────────────────────────────────────────────────────────────
 * Public thresholds (mirror server; safe to expose)
 * ------------------------------------------------------------------------ */
const CONF_HIGH = Number(process.env.NEXT_PUBLIC_CONF_HIGH ?? 0.85);
const CONF_MED = Number(process.env.NEXT_PUBLIC_CONF_MED ?? 0.6);
const AUTO_MAX_SEVERITY = Number(process.env.NEXT_PUBLIC_AUTO_MAX_SEVERITY ?? 2);
const SPEC_MIN_SEVERITY = Number(process.env.NEXT_PUBLIC_SPEC_MIN_SEVERITY ?? 4);
const AUTO_MAX_COST = Number(process.env.NEXT_PUBLIC_AUTO_MAX_COST ?? 1500);
const SPEC_MIN_COST = Number(process.env.NEXT_PUBLIC_SPEC_MIN_COST ?? 5000);
const AUTO_MIN_CONF = Number(process.env.NEXT_PUBLIC_AUTO_MIN_CONF ?? 0.75);

/* ──────────────────────────────────────────────────────────────────────────
 * Small formatting helpers
 * ------------------------------------------------------------------------ */
function band(p?: number) { if (typeof p !== "number") return "Unknown"; return p >= CONF_HIGH ? "High" : p >= CONF_MED ? "Medium" : "Low"; }
function pct(p?: number) { return typeof p === "number" ? `${Math.round(p * 100)}%` : "—"; }
function money(n?: number) { return typeof n === "number" && !Number.isNaN(n) ? `$${n.toLocaleString()}` : "—"; }

function friendlyApiError(raw: string, status?: number) {
  const lower = (raw || "").toLowerCase();
  if (lower.includes("downloading") || lower.includes("fetch") || lower.includes("http")) {
    return "We couldn’t fetch that link. Make sure it’s a direct image URL (JPG/PNG/WebP) and publicly accessible, then try again.";
  }
  if (status && status >= 500) return "Our service hit a hiccup while processing the image. Please try again in a moment.";
  return "We couldn’t process that image or link. Try a different photo or a direct image URL.";
}

/* Weighted aggregation (mirrors server) */
function aggDecisionConf(items: DamageItem[]): number {
  if (!Array.isArray(items) || !items.length) return 0.5;
  let num = 0, den = 0;
  for (const d of items) {
    const sev = Number(d?.severity ?? 1);
    const conf = Number(d?.confidence ?? 0.5);
    const w = 1 + 0.2 * (sev - 1);
    num += conf * w; den += w;
  }
  return den ? num / den : 0.5;
}

function isYoloBoxRel(v: unknown): v is YoloBoxRel {
  if (!v || typeof v !== "object") return false;
  const o = v as Record<string, unknown>;
  const bbox = o["bbox_rel"];
  const conf = o["confidence"];
  return Array.isArray(bbox) && bbox.length === 4 && bbox.every((n) => typeof n === "number") && typeof conf === "number";
}

/* ──────────────────────────────────────────────────────────────────────────
 * Image utils (client-only)
 * ------------------------------------------------------------------------ */
async function compress(file: File, maxW = 1600, quality = 0.72): Promise<File> {
  const img = document.createElement("img");
  const reader = new FileReader();
  const loaded = new Promise<void>((resolve, reject) => {
    reader.onload = () => { img.src = reader.result as string; img.onload = () => resolve(); img.onerror = reject; };
    reader.onerror = reject;
  });
  reader.readAsDataURL(file);
  await loaded;

  const scale = Math.min(1, maxW / img.width);
  const w = Math.round(img.width * scale), h = Math.round(img.height * scale);
  const canvas = document.createElement("canvas"); canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d")!; ctx.drawImage(img, 0, 0, w, h);
  const blob: Blob = await new Promise((res) => canvas.toBlob((b) => res(b!), "image/jpeg", quality));
  return new File([blob], "upload.jpg", { type: "image/jpeg" });
}
function fileToDataUrl(f: File) {
  const r = new FileReader();
  return new Promise<string>((resolve, reject) => { r.onload = () => resolve(r.result as string); r.onerror = reject; r.readAsDataURL(f); });
}
function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image(); img.crossOrigin = "anonymous";
    img.onload = () => resolve(img); img.onerror = reject; img.src = src;
  });
}

/* ──────────────────────────────────────────────────────────────────────────
 * Overlay drawing
 * ------------------------------------------------------------------------ */
function sevColor(sev: number) {
  if (sev >= 5) return "#dc2626"; // red
  if (sev === 4) return "#f97316"; // orange
  if (sev === 3) return "#eab308"; // yellow
  if (sev === 2) return "#22c55e"; // green
  return "#10b981";               // teal
}
function label(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string) {
  if (!text) return;
  ctx.save();
  ctx.font = "12px system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  const pad = 3, w = ctx.measureText(text).width + pad * 2, h = 16;
  ctx.fillRect(x - w / 2, y - 2, w, h);
  ctx.fillStyle = color; ctx.fillText(text, x, y);
  ctx.restore();
}

/** YOLO overlay that matches <img> with object-contain + letterboxing */
function YoloCanvasOverlay({
  imgRef, boxes, show,
}: { imgRef: React.RefObject<HTMLImageElement>; boxes: YoloBoxRel[]; show: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;

    // Position canvas exactly over the <img> element box
    const parent = canvas.parentElement as HTMLElement | null;
    const imgRect = img.getBoundingClientRect();
    const parentRect = parent?.getBoundingClientRect();

    const left = parentRect ? imgRect.left - parentRect.left : 0;
    const top  = parentRect ? imgRect.top  - parentRect.top  : 0;
    const cssW = imgRect.width;
    const cssH = imgRect.height;

    canvas.style.position = "absolute";
    canvas.style.left = `${left}px`;
    canvas.style.top = `${top}px`;
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    canvas.style.pointerEvents = "none";

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // CLEAR
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!show || cssW <= 0 || cssH <= 0) return;

    // LETTERBOX COMPENSATION (object-contain)
    const naturalW = img.naturalWidth || cssW;
    const naturalH = img.naturalHeight || cssH;
    const scale = Math.min(cssW / naturalW, cssH / naturalH);
    const renderedW = naturalW * scale;
    const renderedH = naturalH * scale;
    const offsetX = (cssW - renderedW) / 2;
    const offsetY = (cssH - renderedH) / 2;

    // Draw in CSS pixel space; account for HiDPI
    ctx.scale(dpr, dpr);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#10b981";
    ctx.fillStyle = "rgba(16,185,129,0.2)";
    ctx.font = "12px system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
    ctx.textBaseline = "top";

    boxes.forEach(({ bbox_rel: [nx, ny, nw, nh], confidence }) => {
      const x = offsetX + nx * renderedW;
      const y = offsetY + ny * renderedH;
      const w = nw * renderedW;
      const h = nh * renderedH;

      ctx.beginPath();
      ctx.rect(x, y, w, h);
      ctx.fill();
      ctx.stroke();

      // label pill
      const label = `${Math.round(confidence * 100)}%`;
      const pad = 4, lh = 16;
      const tw = ctx.measureText(label).width + pad * 2;
      const ly = Math.max(0, y - lh - 2);
      ctx.fillStyle = "rgba(0,0,0,0.75)";
      ctx.fillRect(x, ly, tw, lh);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x + pad, ly + 2);

      ctx.fillStyle = "rgba(16,185,129,0.2)";
    });

    const ro = new ResizeObserver(() => {
      canvas.style.width = `${img.getBoundingClientRect().width}px`;
      requestAnimationFrame(() => {});
    });
    ro.observe(img);

    return () => {
      ro.disconnect();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    };
  }, [imgRef, boxes, show]);

  return <canvas ref={canvasRef} aria-hidden="true" />;
}

async function overlaySnapshot(src: string, items: DamageItem[], targetW = 1200) {
  const img = await loadImage(src);
  const scale = Math.min(1, targetW / img.naturalWidth);
  const w = Math.round(img.naturalWidth * scale), h = Math.round(img.naturalHeight * scale);
  const canvas = document.createElement("canvas"); canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, w, h); ctx.drawImage(img, 0, 0, w, h);
  ctx.lineWidth = 2;

  items.forEach((d) => {
    const color = sevColor(Number(d.severity ?? 1));
    ctx.strokeStyle = color; ctx.fillStyle = color + "33";
    if (Array.isArray(d.polygon_rel) && d.polygon_rel.length >= 3) {
      const pts = d.polygon_rel as [number, number][];
      ctx.beginPath();
      pts.forEach(([nx, ny], i) => { const x = nx * w, y = ny * h; if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
      ctx.closePath(); ctx.fill(); ctx.stroke();
      const cx = (pts.reduce((s, p) => s + p[0], 0) / pts.length) * w;
      const cy = (pts.reduce((s, p) => s + p[1], 0) / pts.length) * h;
      label(ctx, String(d.part ?? ""), cx, cy, color);
    } else if (Array.isArray(d.bbox_rel) && d.bbox_rel.length === 4) {
      const [nx, ny, nw, nh] = d.bbox_rel as [number, number, number, number];
      const x = nx * w, y = ny * h, ww = nw * w, hh = nh * h;
      ctx.beginPath(); ctx.rect(x, y, ww, hh); ctx.fill(); ctx.stroke();
      label(ctx, String(d.part ?? ""), x + ww / 2, y + 14, color);
    }
  });
  return canvas.toDataURL("image/jpeg", 0.92);
}

// Helper for print logic when only YOLO boxes are present
async function overlaySnapshotYolo(src: string, boxes: YoloBoxRel[], targetW = 1200) {
  const img = await loadImage(src);
  const scale = Math.min(1, targetW / img.naturalWidth);
  const w = Math.round(img.naturalWidth * scale), h = Math.round(img.naturalHeight * scale);
  const canvas = document.createElement("canvas"); canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, w, h); ctx.drawImage(img, 0, 0, w, h);

  ctx.lineWidth = 2; ctx.strokeStyle = "#10b981"; ctx.fillStyle = "rgba(16,185,129,0.2)";
  boxes.forEach(({ bbox_rel: [nx, ny, nw, nh], confidence }) => {
    const x = nx * w, y = ny * h, ww = nw * w, hh = nh * h;
    ctx.beginPath(); ctx.rect(x, y, ww, hh); ctx.fill(); ctx.stroke();

    const label = `${Math.round(confidence * 100)}%`;
    const pad = 4, lh = 16, tw = ctx.measureText(label).width + pad * 2;
    ctx.fillStyle = "rgba(0,0,0,0.75)";
    ctx.fillRect(x, Math.max(0, y - lh - 2), tw, lh);
    ctx.fillStyle = "#fff"; ctx.font = "12px system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
    ctx.textBaseline = "top"; ctx.fillText(label, x + pad, Math.max(0, y - lh - 2) + 2);
    ctx.fillStyle = "rgba(16,185,129,0.2)";
  });

  return canvas.toDataURL("image/jpeg", 0.92);
}

/* ──────────────────────────────────────────────────────────────────────────
 * Tiny icons
 * ------------------------------------------------------------------------ */
function CopyIcon(props: React.SVGProps<SVGSVGElement>) { return (
  <svg viewBox="0 0 24 24" aria-hidden="true" {...props}>
    <path d="M9 9.5A2.5 2.5 0 0 1 11.5 7H17a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-5.5a2.5 2.5 0 0 1-2.5-2.5v-7Z" fill="none" stroke="currentColor" strokeWidth="1.8"/>
    <path d="M7 15.5V6a2 2 0 0 1 2-2h6.5" fill="none" stroke="currentColor" strokeWidth="1.8"/>
  </svg>
); }
function CheckIcon(props: React.SVGProps<SVGSVGElement>) { return (
  <svg viewBox="0 0 24 24" aria-hidden="true" {...props}>
    <path d="M20 6L9 17l-5-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
); }

/* ──────────────────────────────────────────────────────────────────────────
 * Presentational primitives
 * ------------------------------------------------------------------------ */
function Checkbox({ checked, onChange, label, id }: { checked: boolean; onChange: (v: boolean) => void; label: string; id: string; }) {
  return (
    <label htmlFor={id} className="flex items-center gap-2 text-sm text-slate-800 select-none">
      <input id={id} type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)}
        className="h-4 w-4 accent-indigo-600 transition-transform duration-150 ease-out focus-visible:ring-2 focus-visible:ring-indigo-500 rounded" />
      {label}
    </label>
  );
}

/* ──────────────────────────────────────────────────────────────────────────
 * Damage summary (human-readable)
 * ------------------------------------------------------------------------ */
function buildSummary(result: AnalyzePayload): string {
  const items: DamageItem[] = Array.isArray(result?.damage_items) ? result.damage_items : [];
  if (!items.length) return result?.narrative || "No visible damage detected.";

  const humanJoin = (arr: string[]) =>
    arr.length <= 1 ? (arr[0] || "") : arr.length === 2
      ? `${arr[0]} and ${arr[1]}`
      : `${arr.slice(0, -1).join(", ")}, and ${arr[arr.length - 1]}`;

  const phrases = items.map((d) => {
    const sevText = d.severity >= 5 ? "severe" : d.severity === 4 ? "major" : d.severity === 3 ? "moderate" : "minor";
    const typeMap: Record<string, string> = {
      dent: "dent", scratch: "surface scratch", crack: "structural crack", "paint-chips": "paint chipping",
      broken: "broken component", bent: "bent panel", missing: "missing component", "glass-crack": "glass fracture", unknown: "unspecified damage",
    };
    const typeText = typeMap[d.damage_type] || d.damage_type;
    const zonePart = [d.zone, d.part].filter(Boolean).join(" ");
    const likely = (Array.isArray(d.likely_parts) ? d.likely_parts : []).map(String).filter(p => p && !/paint/i.test(p));
    return `${sevText} ${typeText} on the ${zonePart}${d.needs_paint ? " requiring repainting" : ""}${
      likely.length ? ` with possible replacement of ${humanJoin(Array.from(new Set(likely)))}` : ""}`;
  });

  const joined = phrases.length > 1 ? `${phrases.slice(0, -1).join("; ")}, and ${phrases.slice(-1)}` : phrases[0];
  return `The inspection identified ${joined}. Based on severity and visibility, professional repair is recommended to restore the vehicle to safe operating condition.`;
}

/* ──────────────────────────────────────────────────────────────────────────
 * Sorting helpers
 * ------------------------------------------------------------------------ */
type SortKey = "severity" | "confidence";
type SortDir = "asc" | "desc";

function SortHeader({ label, active, dir, onAsc, onDesc, className = "" }: {
  label: string; active: boolean; dir: SortDir; onAsc: () => void; onDesc: () => void; className?: string;
}) {
  const base = "px-1.5 py-0.5 rounded border text-[11px] transition-colors";
  const on = "border-slate-800 text-slate-900 bg-white/70";
  const off = "border-slate-300 text-slate-600 hover:text-slate-900 hover:bg-white/60";
  return (
    <th className={`p-2 text-left select-none ${className}`}>
      <div className="flex items-center gap-2">
        <span className="text-sm text-slate-700">{label}</span>
        <div className="flex items-center gap-1">
          <button type="button" onClick={onDesc} aria-label={`Sort ${label} descending`} className={`${base} ${active && dir === "desc" ? on : off}`}>▼</button>
          <button type="button" onClick={onAsc} aria-label={`Sort ${label} ascending`} className={`${base} ${active && dir === "asc" ? on : off}`}>▲</button>
        </div>
      </div>
    </th>
  );
}

/* ──────────────────────────────────────────────────────────────────────────
 * Damage table
 * ------------------------------------------------------------------------ */
function DamageTable(props: {
  rows: DamageItem[];
  sortKey: SortKey | null;
  sortDir: SortDir;
  onSetSort: (k: SortKey, dir: SortDir) => void;
  fPaint: "all" | "yes" | "no"; setFPaint: (v: "all" | "yes" | "no") => void;
  fConfMin: string; setFConfMin: (v: string) => void;
  fSearch: string; setFSearch: (v: string) => void;
  onResetFilters: () => void;
}) {
  const { rows, sortKey, sortDir, onSetSort, fPaint, setFPaint, fConfMin, setFConfMin, fSearch, setFSearch, onResetFilters } = props;

  return (
    <div className="rounded-2xl border border-white/30 bg-white/60 backdrop-blur-xl p-5 shadow-[0_8px_30px_rgb(0,0,0,0.06)]">
      <div className="mb-1 flex items-center justify-between gap-3">
        <div className="text-sm font-medium text-slate-900">Detected Damage</div>
        <div className="flex items-center gap-2">
          <input value={fSearch} onChange={(e) => setFSearch(e.target.value)} placeholder="Search zone/part/type…"
            className="w-40 rounded-lg border border-slate-200 bg-white/70 px-2 py-1 text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500" />
          <button type="button" onClick={onResetFilters}
            className="rounded-lg border border-slate-300 bg-white/70 px-2 py-1 text-xs hover:bg-white/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500">
            Reset
          </button>
        </div>
      </div>

      <div className="mb-3 flex items-center justify-end gap-2">
        <select value={fPaint} onChange={(e) => setFPaint(e.target.value as "all" | "yes" | "no")}
          className="w-28 rounded-lg border border-slate-200 bg-white/70 px-2 py-1 text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500">
          <option value="all">Paint: All</option>
          <option value="yes">Paint: Yes</option>
          <option value="no">Paint: No</option>
        </select>
        <input value={fConfMin} onChange={(e) => setFConfMin(e.target.value)} placeholder="Conf ≥ %" inputMode="numeric"
          className="w-20 rounded-lg border border-slate-200 bg-white/70 px-2 py-1 text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500" />
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-[13px] leading-5 border-collapse">
          <thead className="bg-white/60 backdrop-blur text-slate-700">
            <tr>
              <th className="p-2 text-left w-24">Zone</th>
              <th className="p-2 text-left w-24">Part</th>
              <th className="p-2 text-left w-28">Type</th>
              <SortHeader label="Sev" active={sortKey === "severity"} dir={sortDir} onAsc={() => onSetSort("severity", "asc")} onDesc={() => onSetSort("severity", "desc")} className="w-20"/>
              <th className="p-2 text-left w-16">Paint</th>
              <SortHeader label="Conf" active={sortKey === "confidence"} dir={sortDir} onAsc={() => onSetSort("confidence", "asc")} onDesc={() => onSetSort("confidence", "desc")} className="w-32"/>
            </tr>
          </thead>
          <tbody>
            {rows.map((d, i) => (
              <tr key={i} className="border-b border-slate-200/60 last:border-none align-top hover:bg-white/60">
                <td className="p-2">{d.zone}</td>
                <td className="p-2">{d.part}</td>
                <td className="p-2">{d.damage_type}</td>
                <td className="p-2">{d.severity}</td>
                <td className="p-2">{d.needs_paint ? "Yes" : "No"}</td>
                <td className="p-2">{pct(d.confidence)} ({band(d.confidence)})</td>
              </tr>
            ))}
            {!rows.length && (
              <tr><td colSpan={6} className="p-3 text-center text-sm text-slate-500">No rows match your filters.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ──────────────────────────────────────────────────────────────────────────
 * Decision explainer helpers
 * ------------------------------------------------------------------------ */
function sevClass(sev: number) {
  if (sev >= SPEC_MIN_SEVERITY) return "border-rose-200 bg-rose-50 text-rose-700";
  if (sev <= AUTO_MAX_SEVERITY) return "border-emerald-200 bg-emerald-50 text-emerald-700";
  return "border-amber-200 bg-amber-50 text-amber-700";
}
function costClass(cost?: number) {
  if (typeof cost !== "number") return "border-slate-200 bg-white/70 text-slate-700";
  if (cost >= SPEC_MIN_COST) return "border-rose-200 bg-rose-50 text-rose-700";
  if (cost <= AUTO_MAX_COST) return "border-emerald-200 bg-emerald-50 text-emerald-700";
  return "border-amber-200 bg-amber-50 text-amber-700";
}
function confClass(conf: number) { return conf >= AUTO_MIN_CONF ? "border-emerald-200 bg-emerald-50 text-emerald-700" : "border-amber-200 bg-amber-50 text-amber-700"; }

function whyBlurb(label: "AUTO-APPROVE" | "INVESTIGATE" | "SPECIALIST", m: { maxSev: number; costHigh?: number; aggConf: number; }) {
  if (label === "SPECIALIST") {
    if (m.maxSev >= SPEC_MIN_SEVERITY && (m.costHigh ?? 0) >= SPEC_MIN_COST) return "Escalated because severity and cost exceed specialist thresholds.";
    if (m.maxSev >= SPEC_MIN_SEVERITY) return `Escalated because severity ≥ ${SPEC_MIN_SEVERITY}.`;
    if ((m.costHigh ?? 0) >= SPEC_MIN_COST) return `Escalated because cost ≥ ${SPEC_MIN_COST.toLocaleString("en-US",{style:"currency",currency:"USD"})}.`;
    return "Escalated based on policy thresholds.";
  }
  if (label === "AUTO-APPROVE") return "All checks are within auto-approve thresholds.";
  const blockers: string[] = [];
  if (m.maxSev > AUTO_MAX_SEVERITY) blockers.push(`severity > ${AUTO_MAX_SEVERITY}`);
  if ((m.costHigh ?? 0) > AUTO_MAX_COST) blockers.push(`cost > $${AUTO_MAX_COST}`);
  if (m.aggConf < AUTO_MIN_CONF) blockers.push(`confidence < ${Math.round(AUTO_MIN_CONF * 100)}%`);
  return blockers.length ? `Needs review: ${blockers.join(", ")}.` : "Needs review.";
}

/* ──────────────────────────────────────────────────────────────────────────
 * Main Page
 * ------------------------------------------------------------------------ */
export default function Home() {
  // Inputs
  const [mode, setMode] = useState<"upload" | "url">("upload");
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState("");
  const [preview, setPreview] = useState("");

  // Network/result
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalyzePayload | null>(null);
  const [error, setError] = useState("");
  const [validationIssues, setValidationIssues] = useState<string[] | null>(null);

  // YOLO overlay + detect debug
  const [yoloBoxes, setYoloBoxes] = useState<YoloBoxRel[]>([]);
  const [detectIssues, setDetectIssues] = useState<string[] | null>(null);
  const [yoloDebug, setYoloDebug] = useState<RoboflowDebug | null>(null);

  // Toggles
  const [showOverlay, setShowOverlay] = useState(true);
  const [showAudit, setShowAudit] = useState(false);
  const [showWhy, setShowWhy] = useState(false);
  const [showCost, setShowCost] = useState(false);

  // Copy states
  const [copiedSummary, setCopiedSummary] = useState(false);
  const [copiedEstimate, setCopiedEstimate] = useState(false);

  // Print
  const [snapshotUrl, setSnapshotUrl] = useState("");

  // Refs
  const imgRef = useRef<HTMLImageElement>(null);

  const switchMode = useCallback((next: "upload" | "url") => {
    setMode(next); setResult(null); setError(""); setValidationIssues(null); setPreview("");
    if (next === "upload") setImageUrl(""); else setFile(null);
  }, []);

  const onFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    setResult(null); setError(""); setValidationIssues(null); setFile(f); setPreview(f ? URL.createObjectURL(f) : "");
  }, []);

  const onUrlChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value; setImageUrl(val); setResult(null); setError(""); setValidationIssues(null); setPreview(val || "");
  }, []);

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true); setError(""); setValidationIssues(null); setResult(null); setYoloBoxes([]); setDetectIssues(null); setYoloDebug(null);

    try {
      const analyzeForm = new FormData(), detectForm = new FormData();

      if (mode === "upload") {
        if (!file) { setError("Please choose a photo to analyze."); return; }
        const compressed = await compress(file);
        const dataUrl = await fileToDataUrl(compressed);
        detectForm.append("file", compressed);
        analyzeForm.append("file", compressed);
        analyzeForm.append("image_data_url", dataUrl); // optional server-side use
      } else {
        const raw = imageUrl.trim();
        if (!raw) { setError("Please paste an image link to analyze."); return; }
        detectForm.append("imageUrl", raw);
        analyzeForm.append("imageUrl", raw);
      }

      // 1) DETECT
      const dr = await fetch("/api/detect", { method: "POST", body: detectForm });
      if (!dr.ok) { setError(friendlyApiError(await dr.text(), dr.status)); return; }
      const detectJson = await dr.json();

      // Capture debug
      if (detectJson?.yolo_debug) setYoloDebug(detectJson.yolo_debug as RoboflowDebug);

      if (detectJson && detectJson.is_vehicle === false) {
        setError("We couldn’t detect a vehicle in that image. Please upload a photo that clearly shows a vehicle.");
        return;
      }

      const boxes = Array.isArray(detectJson?.yolo_boxes)
        ? (detectJson.yolo_boxes as unknown[]).filter(isYoloBoxRel) as YoloBoxRel[]
        : [];
      setYoloBoxes(boxes);

      if (Array.isArray(detectJson?.issues) && detectJson.issues.length) setDetectIssues(detectJson.issues.map(String));

      // 2) ANALYZE – pass YOLO boxes as seeds if present
      const seeds: { bbox_rel: [number, number, number, number]; confidence: number }[] =
        Array.isArray(detectJson?.yolo_boxes)
          ? (detectJson.yolo_boxes as unknown[])
              .filter(isYoloBoxRel)
              .map((b) => ({ bbox_rel: b.bbox_rel, confidence: typeof b.confidence === "number" ? b.confidence : 0.5 }))
          : [];
      analyzeForm.append("yolo", JSON.stringify(seeds));

      const ar = await fetch("/api/analyze", { method: "POST", body: analyzeForm });
      if (!ar.ok) { setError(friendlyApiError(await ar.text(), ar.status)); return; }
      const j = await ar.json();

      // Avoid false positives on non-vehicle photos
      const noItems = !Array.isArray(j?.damage_items) || j.damage_items.length === 0;
      if (noItems && Number(j?.vehicle?.confidence ?? 0) < 0.15) {
        setError("We couldn't detect a vehicle in that image. Please upload a photo that clearly shows a vehicle.");
        return;
      }

      setResult(j);
      if (Array.isArray(detectJson?.issues) && detectJson.issues.length) setValidationIssues(detectJson.issues);
    } catch {
      setError("We couldn’t process that image or link. Try a different photo or a direct image URL.");
    } finally {
      setLoading(false);
    }
  }, [file, imageUrl, mode]);

  // Vehicle meta
  const make = result?.vehicle?.make ?? "—";
  const model = result?.vehicle?.model ?? "—";
  const color = result?.vehicle?.color ?? "—";

  // Sorting & filters
  const [sortKey, setSortKey] = useState<SortKey | null>(null);
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [fPaint, setFPaint] = useState<"all" | "yes" | "no">("all");
  const [fConfMin, setFConfMin] = useState("");
  const [fSearch, setFSearch] = useState("");

  const setSort = useCallback((k: SortKey, dir: SortDir) => { setSortKey(k); setSortDir(dir); }, []);
  const resetFilters = useCallback(() => { setFPaint("all"); setFConfMin(""); setFSearch(""); }, []);

  const filteredSorted: DamageItem[] = useMemo(() => {
    const items: DamageItem[] = Array.isArray(result?.damage_items) ? result.damage_items : [];
    const q = fSearch.trim().toLowerCase();
    const confMin = fConfMin ? Number(fConfMin) / 100 : undefined;

    const filtered = items.filter((d) => {
      const paintOk = fPaint === "all" || (fPaint === "yes" && d.needs_paint) || (fPaint === "no" && !d.needs_paint);
      const confOk = confMin === undefined || Number(d.confidence ?? 0) >= confMin;
      const hay = `${d.zone} ${d.part} ${d.damage_type}`.toLowerCase();
      return paintOk && confOk && (!q || hay.includes(q));
    });

    if (!sortKey) return filtered;
    const dirMul = sortDir === "asc" ? 1 : -1;
    return [...filtered].sort((a, b) => ((Number(a?.[sortKey] ?? 0) - Number(b?.[sortKey] ?? 0)) * dirMul));
  }, [result, fPaint, fConfMin, fSearch, sortKey, sortDir]);

  const decisionConf = useMemo(() =>
    aggDecisionConf(Array.isArray(result?.damage_items) ? result.damage_items : []), [result]);

  const copyToClipboard = useCallback(async (text: string, which: "summary" | "estimate") => {
    try {
      await navigator.clipboard.writeText(text);
      (which === "summary" ? setCopiedSummary : setCopiedEstimate)(true);
      setTimeout(() => (which === "summary" ? setCopiedSummary : setCopiedEstimate)(false), 1200);
    } catch {}
  }, []);

  const handlePrint = useCallback(async () => {
    try {
      if (preview) {
        if (yoloBoxes.length > 0) {
          try { setSnapshotUrl(await overlaySnapshotYolo(preview, yoloBoxes)); }
          catch { setSnapshotUrl(preview); }
        } else if (result?.damage_items?.length) {
          try { setSnapshotUrl(await overlaySnapshot(preview, result.damage_items)); }
          catch { setSnapshotUrl(preview); }
        } else {
          setSnapshotUrl(preview);
        }
        setTimeout(() => window.print(), 50);
        return;
      }
    } catch {}
    window.print();
  }, [preview, result, yoloBoxes]);

  const samples = [
    { url: "https://preview.redd.it/could-this-side-impact-damage-possibly-be-a-structural-v0-48xk2ye4dad81.jpg?width=640&crop=smart&auto=webp&s=2e8661fe071e2aacbd413ca4821aab837859176c" },
    { url: "https://diminishedvalueofgeorgia.com/wp-content/uploads/Evidence-of-Severe-Front-End-Damage-on-a-Vehicle-After-an-Accident-scaled.jpg" },
    { url: "https://i.redd.it/902pxt9a8r4c1.jpg" },
    { url: "https://brightonpanelworks.com.au/wp-content/uploads/2019/10/auto-3734396_1280.jpg" },
  ];

  const selectSample = useCallback((url: string) => {
    setMode("url"); setImageUrl(url); setPreview(url); setResult(null); setError(""); setValidationIssues(null);
  }, []);

  return (
    <main className="relative min-h-screen text-slate-900">
      {/* Background */}
      <div className="fixed inset-0 -z-10 bg-gradient-to-br from-slate-50 via-indigo-50 to-sky-50 bg-[length:200%_200%]" />
      <div className="bg-animated fixed inset-0 -z-10" />
      <div className="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(ellipse_at_top_right,rgba(29,78,216,0.06),transparent_55%),radial-gradient(ellipse_at_bottom_left,rgba(2,132,199,0.06),transparent_55%)]" />

      <style jsx global>{`
        @media print {
          @page { size: A4 portrait; margin: 0.5in; }
          html, body { background: #fff !important; }
          .print-break { page-break-before: always; }
        }
        .bg-animated {
          animation: gradientShift 18s ease-in-out infinite;
          background: linear-gradient(120deg, rgba(99,102,241,.08), rgba(14,165,233,.08), rgba(99,102,241,.08));
          background-size: 200% 200%;
        }
        @keyframes gradientShift {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
      `}</style>

      {/* Header */}
      <header className="border-b border-white/30 bg-white/60 backdrop-blur-xl print:hidden shadow-[0_4px_20px_rgba(0,0,0,0.05)]">
        <div className="mx-auto max-w-7xl px-6 py-5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h1 className="text-xl sm:text-2xl font-semibold tracking-tight">Car Damage Estimator</h1>
            <p className="text-sm text-slate-600">Upload a vehicle photo or paste an image URL to generate a structured damage report, cost band, and routing decision.</p>
          </div>
          <div className="flex items-center gap-2">
            <button type="button" onClick={handlePrint} disabled={!result}
              className="inline-flex items-center justify-center rounded-lg border border-slate-300 bg-white/70 px-3 py-2 text-sm font-medium disabled:opacity-50 hover:shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
              title={result ? "Export Report PDF" : "Run an analysis first"}>
              Export Report PDF
            </button>
          </div>
        </div>
      </header>

      {/* Layout */}
      <div className="mx-auto max-w-7xl px-6 py-6 grid gap-6 lg:grid-cols-12">
        {/* Left column */}
        <section className="lg:col-span-4 print:hidden">
          <div className="lg:sticky lg:top-6 space-y-4">
            {/* Input */}
            <form onSubmit={handleSubmit} className="rounded-2xl border border-white/30 bg-white/60 backdrop-blur-xl p-4 shadow-[0_8px_30px_rgb(0,0,0,0.06)] space-y-3">
              <div className="flex rounded-lg overflow-hidden border border-slate-200/70">
                <button type="button" onClick={() => switchMode("upload")}
                        className={`flex-1 px-3 py-2 text-sm ${mode === "upload" ? "bg-indigo-600 text-white" : "bg-white/70 text-slate-700 hover:bg-white/90"}`}>Upload</button>
                <button type="button" onClick={() => switchMode("url")}
                        className={`flex-1 px-3 py-2 text-sm ${mode === "url" ? "bg-indigo-600 text-white" : "bg-white/70 text-slate-700 hover:bg-white/90"}`}>URL</button>
              </div>

              {mode === "upload" ? (
                <label className="text-sm font-medium block">
                  <span className="sr-only">Upload image</span>
                  <input type="file" accept="image/*" onChange={onFile}
                    className="block w-full rounded-lg border border-slate-200 bg-white/70 px-3 py-2 text-sm file:mr-3 file:rounded file:border-0 file:bg-slate-100 file:px-3 file:py-2 file:text-sm hover:file:bg-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500" />
                </label>
              ) : (
                <input type="url" placeholder="https://example.com/damaged-car.jpg" value={imageUrl} onChange={onUrlChange}
                  className="w-full rounded-lg border border-slate-200 bg-white/70 px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500" />
              )}

              <button type="submit" disabled={loading || (mode === "upload" ? !file : !imageUrl)}
                className="inline-flex items-center justify-center rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-60 hover:bg-indigo-700 hover:shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500">
                {loading ? "Analyzing…" : "Analyze"}
              </button>

              {validationIssues && (
                <div className="rounded-lg border border-amber-200/70 bg-amber-50/80 p-3 text-sm text-amber-800">
                  <div className="font-medium mb-1">Note: accuracy of this report may be affected by:</div>
                  <ul className="list-disc pl-5">{validationIssues.map((v, i) => (<li key={i}>{String(v)}</li>))}</ul>
                </div>
              )}

              <div aria-live="polite">
                {error && (<div className="rounded-lg border border-rose-200/70 bg-rose-50/80 p-3 text-sm text-rose-700">{String(error)}</div>)}
              </div>

              <div className="pt-1 flex flex-col gap-2">
                <Checkbox id="overlay" checked={showOverlay} onChange={setShowOverlay} label="Show damage overlay" />
                <Checkbox id="audit" checked={showAudit} onChange={setShowAudit} label="Show audit metadata" />
              </div>
            </form>

            {/* Image */}
            <div className="rounded-2xl border border-white/30 bg-white/60 backdrop-blur-xl p-3 shadow-[0_8px_30px_rgb(0,0,0,0.06)]">
              <div className="mb-2 text-sm font-medium">Image Preview</div>
              <div className="relative">
                {preview ? (
                  <>
                    <img ref={imgRef} src={preview} alt="preview" className="w-full rounded-lg border border-slate-200 max-h-[360px] object-contain bg-slate-50" />
                    {showOverlay && (
                      <YoloCanvasOverlay imgRef={imgRef as React.RefObject<HTMLImageElement>} boxes={yoloBoxes} show={showOverlay} />
                    )}
                  </>
                ) : (
                  <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-slate-300 text-sm text-slate-500">
                    {mode === "upload" ? "Upload a photo to begin" : "Paste an image URL to preview"}
                  </div>
                )}
              </div>

              {/* Overlay source + YOLO debug */}
              <div className="mt-2 flex items-center justify-between">
                <span className="text-[11px] rounded border border-emerald-300 bg-emerald-50 px-2 py-0.5 text-emerald-700">
                  overlay: YOLO ({yoloBoxes.length})
                </span>

                <details className="text-[12px]">
                  <summary className="cursor-pointer text-indigo-700 hover:text-indigo-900">YOLO debug</summary>
                  <div className="mt-2 space-y-2">
                    {/* Server-side env and HTTP hints */}
                    {yoloDebug ? (
                      <div className="space-y-1">
                        {Array.isArray(yoloDebug.missing_env) && yoloDebug.missing_env.length > 0 ? (
                          <div className="text-rose-700">
                            <div className="font-medium">Missing env vars:</div>
                            <ul className="ml-5 list-disc">
                              {yoloDebug.missing_env.map((k: string) => <li key={k}>{k}</li>)}
                            </ul>
                          </div>
                        ) : (
                          <div className="text-slate-700">
                            <div><span className="text-slate-500">HTTP status:</span> {yoloDebug.status ?? "—"} {yoloDebug.ok === false ? "(not ok)" : yoloDebug.ok === true ? "(ok)" : ""}</div>
                            <div><span className="text-slate-500">Parse path:</span> {yoloDebug.parse_path ?? "—"}</div>
                            <div><span className="text-slate-500">Parsed count:</span> {typeof yoloDebug.parsed_count === "number" ? yoloDebug.parsed_count : "—"}</div>
                            {yoloDebug.error && <div className="text-amber-700">Error: {String(yoloDebug.error)}</div>}
                            {yoloDebug.body_snippet && (
                              <div className="mt-1">
                                <div className="text-slate-500">Body (first 240 chars):</div>
                                <pre className="whitespace-pre-wrap break-words rounded border bg-white/60 p-2 text-[11px]">{yoloDebug.body_snippet}</pre>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-slate-500">No server debug info.</div>
                    )}

                    {/* Client-side detect issues */}
                    {detectIssues?.length ? (
                      <div className="text-amber-700">
                        <div className="font-medium">Detect issues:</div>
                        <ul className="list-disc ml-5">
                          {detectIssues.map((v, i) => <li key={i}>{v}</li>)}
                        </ul>
                      </div>
                    ) : (
                      <div className="text-slate-500">No detect issues reported.</div>
                    )}

                    {/* Boxes list */}
                    <div className="text-slate-700">
                      <div className="font-medium">Boxes (first 5):</div>
                      {yoloBoxes.length ? (
                        <ul className="ml-5 list-disc">
                          {yoloBoxes.slice(0, 5).map((b, i) => (
                            <li key={i}>
                              bbox_rel=[{b.bbox_rel.map(n => n.toFixed(3)).join(", ")}], conf={Math.round(b.confidence * 100)}%
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <div className="text-slate-500">[] (Roboflow returned no boxes)</div>
                      )}
                    </div>
                  </div>
                </details>
              </div>
            </div>

            {/* Samples */}
            <div className="rounded-2xl border border-white/30 bg-white/60 backdrop-blur-xl p-4 shadow-[0_8px_30px_rgb(0,0,0,0.06)]">
              <div className="mb-2 text-sm font-medium">Sample Images</div>
              <div className="grid grid-cols-4 gap-2">
                {samples.map((s) => (
                  <button key={s.url} type="button" onClick={() => selectSample(s.url)}
                    className="group relative rounded-lg overflow-hidden border border-slate-200 bg-white/60 hover:shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500" title="Sample image">
                    <img src={s.url} alt="Sample" className="h-16 w-full object-cover" />
                  </button>
                ))}
              </div>
              <div className="mt-2 text-[11px] text-slate-500">Tip: Click a sample to auto-fill the URL and preview instantly.</div>
            </div>

            {/* Legend */}
            <div className="rounded-2xl border border-white/30 bg-white/60 backdrop-blur-xl p-5">
              <div className="mb-1 text-sm font-medium">Legend & Settings</div>
              <div className="text-xs text-slate-700 space-y-2">
                <div>
                  <span className="font-medium">Routing:</span>{" "}
                  <span className="inline-flex gap-2 flex-wrap">
                    <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5">AUTO-APPROVE</span>
                    <span className="rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5">INVESTIGATE</span>
                    <span className="rounded-full border border-rose-200 bg-rose-50 px-2 py-0.5">SPECIALIST</span>
                  </span>
                  <div className="mt-1 text-slate-600">Decision uses severity, confidence, and estimated cost thresholds.</div>
                </div>
                <div><span className="font-medium">Severity colors:</span> 1–2 (green) • 3 (yellow) • 4 (orange) • 5 (red)</div>
                <div><span className="font-medium">Confidence bands:</span> High ≥ {Math.round(CONF_HIGH * 100)}% • Medium ≥ {Math.round(CONF_MED * 100)}% • otherwise Low</div>
                <div><span className="font-medium">Cost assumptions:</span> Server-side policy drives labor, paint, parts, and contingency.</div>
              </div>
            </div>
          </div>
        </section>

        {/* Right column */}
        <section className="lg:col-span-8 space-y-4 print:col-span-12">
          {/* Routing decision */}
          {result?.decision && (
            <div className="rounded-2xl border border-white/30 bg-white/60 backdrop-blur-xl p-5">
              <div className="mb-1 text-sm font-medium">Routing Decision</div>
              <div className="flex flex-wrap items-center gap-2">
                <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs ${
                    result.decision.label === "AUTO-APPROVE"
                      ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                      : result.decision.label === "SPECIALIST"
                      ? "border-rose-200 bg-rose-50 text-rose-700"
                      : "border-amber-200 bg-amber-50 text-amber-700"
                }`}>{result.decision.label}</span>
              </div>
              <div className="mt-2 text-xs text-slate-600">Confidence: {pct(decisionConf)} ({band(decisionConf)})</div>

              <div className="mt-3">
                <button type="button" onClick={() => setShowWhy(v => !v)} className="text-xs text-indigo-700 hover:text-indigo-900 underline underline-offset-2">
                  {showWhy ? "Hide details" : "Why this decision?"}
                </button>

                {showWhy && (
                  <div className="mt-2 space-y-3">
                    <div className="text-xs text-slate-700">
                      {whyBlurb(result.decision.label, {
                        maxSev: Math.max(0, ...(result.damage_items ?? []).map(d => Number(d.severity ?? 0))),
                        costHigh: result.estimate?.cost_high,
                        aggConf: decisionConf,
                      })}
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                      <div className={`rounded-xl border px-3 py-2 ${sevClass(Math.max(0, ...(result.damage_items ?? []).map(d => Number(d.severity ?? 0))))}`}>
                        <div className="text-[11px] opacity-80">Severity (need ≤ {AUTO_MAX_SEVERITY})</div>
                        <div className="text-sm font-medium">
                          Max {Math.max(0, ...(result.damage_items ?? []).map(d => Number(d.severity ?? 0)))}
                        </div>
                        <div className="text-[11px] opacity-80 mt-0.5">Escalates at ≥ {SPEC_MIN_SEVERITY}</div>
                      </div>
                      <div className={`rounded-xl border px-3 py-2 ${costClass(result.estimate?.cost_high)}`}>
                        <div className="text-[11px] opacity-80">Cost (need ≤ ${AUTO_MAX_COST})</div>
                        <div className="text-sm font-medium">High {money(result.estimate?.cost_high)}</div>
                        <div className="text-[11px] opacity-80 mt-0.5">Escalates at ≥ ${SPEC_MIN_COST}</div>
                      </div>
                      <div className={`rounded-xl border px-3 py-2 ${confClass(decisionConf)}`}>
                        <div className="text-[11px] opacity-80">Confidence (need ≥ {Math.round(AUTO_MIN_CONF * 100)}%)</div>
                        <div className="text-sm font-medium">Avg {pct(decisionConf)}</div>
                        <div className="text-[11px] opacity-80 mt-0.5">**Doesn’t escalate on its own</div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Vehicle metadata */}
          {result && (
            <div className="rounded-2xl border border-white/30 bg-white/60 backdrop-blur-xl p-5">
              <div className="mb-2 text-sm font-medium">Vehicle metadata</div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-sm">
                <div><span className="text-slate-500">Make:</span> {make}</div>
                <div><span className="text-slate-500">Model:</span> {model}</div>
                <div><span className="text-slate-500">Color:</span> {color}</div>
              </div>
              <div className="mt-1 text-xs text-slate-500">Vehicle confidence: {pct(result?.vehicle?.confidence)} ({band(result?.vehicle?.confidence)})</div>
            </div>
          )}

          {/* Damage table */}
          {Array.isArray(result?.damage_items) && result.damage_items.length > 0 && (
            <DamageTable
              rows={filteredSorted}
              sortKey={sortKey}
              sortDir={sortDir}
              onSetSort={setSort}
              fPaint={fPaint} setFPaint={setFPaint}
              fConfMin={fConfMin} setFConfMin={setFConfMin}
              fSearch={fSearch} setFSearch={setFSearch}
              onResetFilters={resetFilters}
            />
          )}

          {/* Summary with copy */}
          {(result?.narrative || (Array.isArray(result?.damage_items) && result.damage_items.length > 0)) && (
            <div className="rounded-2xl border border-white/30 bg-white/60 backdrop-blur-xl p-5">
              <div className="mb-1 flex items-center justify-between">
                <div className="text-sm font-medium">Damage summary</div>
                <button type="button" onClick={() => copyToClipboard(buildSummary(result as AnalyzePayload), "summary")}
                        className="inline-flex h-7 w-7 items-center justify-center rounded border border-slate-300 bg-white/70 hover:bg-white/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500" title="Copy summary">
                  {copiedSummary ? <CheckIcon className="h-4 w-4" /> : <CopyIcon className="h-4 w-4" />}
                </button>
              </div>
              <div className="text-sm leading-relaxed text-slate-800">{buildSummary(result as AnalyzePayload)}</div>
            </div>
          )}

          {/* Estimate + Cost breakdown */}
          {result?.estimate?.breakdown && (
            <div className="rounded-2xl border border-white/30 bg-white/60 backdrop-blur-xl p-5">
              <div className="mb-1 flex items-center justify-between">
                <div className="text-sm font-medium">Estimated Repair Cost</div>
                <button type="button" onClick={() => copyToClipboard(
                  `${money(result.estimate.cost_low)} – ${money(result.estimate.cost_high)}`, "estimate")}
                        className="inline-flex h-7 w-7 items-center justify-center rounded border border-slate-300 bg-white/70 hover:bg-white/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500" title="Copy estimate">
                  {copiedEstimate ? <CheckIcon className="h-4 w-4" /> : <CopyIcon className="h-4 w-4" />}
                </button>
              </div>

              <div className="text-xl font-semibold tracking-tight">
                {money(result.estimate.cost_low)} – {money(result.estimate.cost_high)}
              </div>
              {Array.isArray(result.estimate.assumptions) && result.estimate.assumptions.length > 0 && (
                <div className="mt-2 text-xs text-slate-500">{result.estimate.assumptions.join(" • ")}</div>
              )}

              <div className="mt-3">
                <button type="button" onClick={() => setShowCost(v => !v)} className="text-xs text-indigo-700 hover:text-indigo-900 underline underline-offset-2">
                  {showCost ? "Hide details" : "Cost breakdown"}
                </button>

                {/* Three-row, no-table explainer */}
                {showCost && (() => {
                  const bd = result.estimate.breakdown!;
                  const assumptions = Array.isArray(result.estimate.assumptions) ? result.estimate.assumptions : [];

                  const numOr = (s: string | undefined, fallback: number) => {
                    const m = s?.match(/\$?(\d+(?:\.\d+)?)/); return m ? Number(m[1]) : fallback;
                  };

                  // Totals to derive true rates from numeric breakdown (exact)
                  const totalHours = bd.lines.reduce((s, l) => s + Number(l.est_labor_hours || 0), 0);

                  // Prefer server-deduped panel count; fallback to line count where paint_cost > 0
                  const paintedPanelCount =
                    typeof bd.paint_units === "number"
                      ? bd.paint_units
                      : bd.lines.filter((l) => (l.paint_cost || 0) > 0).length;

                  // Fallbacks from assumptions (legacy) — used ONLY if derivation is impossible
                  const laborAssump = assumptions.find((a) => /labor/i.test(a));
                  const paintAssump = assumptions.find((a) => /(paint|material)/i.test(a));
                  const LABOR_FALLBACK = numOr(laborAssump, 95);
                  const PAINT_FALLBACK = numOr(paintAssump, 180);

                  // Derive BASE (pre-correction) labor rate & effective panel rate
                  const baseLaborRate =
                    totalHours > 0 && typeof bd.labor_pre_correction === "number"
                      ? Math.round(bd.labor_pre_correction / totalHours)
                      : LABOR_FALLBACK;

                  const laborCorr =
                    typeof bd.labor_correction_factor === "number"
                      ? bd.labor_correction_factor
                      : typeof bd.labor_pre_correction === "number" && bd.labor_pre_correction > 0
                      ? +(bd.labor / bd.labor_pre_correction).toFixed(2)
                      : 1;

                  const effectiveLaborRate = Math.round(baseLaborRate * laborCorr);
                  const panelRate =
                    paintedPanelCount > 0 ? Math.round(bd.paint / paintedPanelCount) : PAINT_FALLBACK;

                  // Parts preview (unique names from items)
                  const partsNames = (result.damage_items ?? []).flatMap(d => {
                    const listed = Array.isArray(d.likely_parts) ? d.likely_parts.map(String) : [];
                    if (listed.length) return listed;
                    return d.severity >= 4 && d.part && d.part !== "unknown" ? [String(d.part)] : [];
                  });
                  const partsUnique = Array.from(new Set(partsNames.map(p => p.toLowerCase())));
                  const partsCount = (bd.parts_detail?.length ?? partsUnique.length) || (bd.parts ? 1 : 0);
                  const perPartAvg = partsCount ? Math.round(bd.parts / partsCount) : 0;

                  const hoursStr = `${totalHours.toFixed(2)} hr${totalHours !== 1 ? "s" : ""} × $${effectiveLaborRate}/hr`;
                  const panelsStr = `${paintedPanelCount || 0} panel${paintedPanelCount === 1 ? "" : "s"} × $${panelRate}/panel`;
                  const partsStr = partsCount
                    ? `${partsCount} part${partsCount === 1 ? "" : "s"}${perPartAvg ? ` ~ $${perPartAvg} each` : ""}`
                    : "No parts estimated";

                  const contingency = Number((bd as any).contingency || 0);
                  const subtotal = bd.labor + bd.paint + bd.parts + contingency;
                  const lo = Number(result.estimate.cost_low || 0);
                  const hi = Number(result.estimate.cost_high || 0);
                  const inBand = subtotal >= lo && subtotal <= hi;

                  return (
                    <div className="mt-3 space-y-3">
                      {/* Row 1: Labor */}
                      <div className="rounded-xl border border-slate-200 bg-white/70 p-3">
                        <div className="text-sm font-medium text-slate-900">Labor</div>
                        <div className="mt-1 text-[13px] text-slate-700">
                          {hoursStr} = <span className="font-medium">${bd.labor.toLocaleString()}</span>
                        </div>
                      </div>

                      {/* Row 2: Paint & Materials */}
                      <div className="rounded-xl border border-slate-200 bg-white/70 p-3">
                        <div className="text-sm font-medium text-slate-900">Paint &amp; Materials</div>
                        <div className="mt-1 text-[13px] text-slate-700">
                          {panelsStr} = <span className="font-medium">${bd.paint.toLocaleString()}</span>
                        </div>
                        {paintedPanelCount > 0 && (
                          <div className="mt-1 text-[12px] text-slate-500">
                            Paint is applied once per affected panel; light blends discounted where applicable.
                          </div>
                        )}
                      </div>

                      {/* Row 3: Parts */}
                      <div className="rounded-xl border border-slate-200 bg-white/70 p-3">
                        <div className="text-sm font-medium text-slate-900">Parts</div>

                        {bd.dynamic_parts_used && Array.isArray(bd.parts_detail) && bd.parts_detail.length > 0 ? (
                          <>
                            <div className="mt-1 text-[13px] text-slate-700">
                              {bd.parts_detail.length} part{bd.parts_detail.length === 1 ? "" : "s"} ={" "}
                              <span className="font-medium">${bd.parts.toLocaleString()}</span>
                              <span className="ml-2 text-[11px] text-slate-500">(dynamic + sanity bands)</span>
                            </div>
                            <ul className="mt-1 text-[12px] text-slate-600 space-y-0.5">
                              {bd.parts_detail.slice(0, 10).map((p, i) => (
                                <li key={i} className="flex items-center justify-between gap-3">
                                  <span className="truncate">{p.name} × {p.qty}</span>
                                  <span className="whitespace-nowrap">
                                    ${p.unit_price} = {(p.line_total).toLocaleString()}
                                  </span>
                                </li>
                              ))}
                              {bd.parts_detail.length > 10 && (
                                <li className="text-[11px] text-slate-500">+{bd.parts_detail.length - 10} more</li>
                              )}
                            </ul>
                          </>
                        ) : (
                          <div className="mt-1 text-[13px] text-slate-700">
                            {partsStr} = <span className="font-medium">${bd.parts.toLocaleString()}</span>
                            <span className="ml-2 text-[11px] text-slate-500">
                              {bd.dynamic_parts_used ? "(dynamic + banded)" : "(baseline + banded)"}
                            </span>
                          </div>
                        )}
                      </div>

                      {/* Row 4: Hidden/Teardown Contingency */}
                      {((bd as any).contingency || 0) > 0 && (
                        <div className="rounded-xl border border-slate-200 bg-white/70 p-3">
                          <div className="text-sm font-medium text-slate-900">Hidden/Teardown Contingency</div>
                          <div className="mt-1 text-[13px] text-slate-700">
                            Applied to labor + parts = <span className="font-medium">${((bd as any).contingency).toLocaleString()}</span>
                          </div>
                        </div>
                      )}

                      {/* Subtotal vs Band */}
                      <div className="flex items-center justify-between mt-2">
                        <div className="text-sm text-slate-600">
                          Range: <span className="font-medium">{money(lo)} – {money(hi)}</span>
                        </div>
                        <div className={`text-lg font-semibold tracking-tight ${inBand ? "text-slate-900" : "text-amber-700"}`}>
                          Subtotal: {money(subtotal)}
                        </div>
                      </div>
                      {!inBand && (
                        <div className="text-[12px] text-amber-700">
                          Note: subtotal is outside the displayed band; variance or inputs may need adjustment.
                        </div>
                      )}
                    </div>
                  );
                })()}
              </div>
            </div>
          )}

          {/* Audit */}
          {result && showAudit && (
            <div className="rounded-2xl border border-white/30 bg-white/60 backdrop-blur-xl p-5">
              <div className="mb-1 text-sm font-medium">Audit Metadata</div>
              <div className="grid grid-cols-1 gap-y-1 text-xs sm:grid-cols-2">
                <div><span className="text-slate-500">Schema:</span> {result.schema_version ?? "—"}</div>
                <div><span className="text-slate-500">Model:</span> {result.model ?? "—"}</div>
                <div><span className="text-slate-500">runId:</span> {result.runId ?? "—"}</div>
                <div className="truncate"><span className="text-slate-500">image_sha256:</span> {result.image_sha256 ?? "—"}</div>
              </div>
            </div>
          )}

          {/* Print-only image */}
          <div className="hidden print:block">
            <h2 className="text-xl font-semibold">Car Damage Report</h2>
            <div className="text-xs text-slate-600">Generated: {new Date().toLocaleString()}</div>
            <hr className="my-2" />
            {(snapshotUrl || preview) && (
              <img src={snapshotUrl || preview} alt="Vehicle image" className="w-full h-auto border rounded mb-4" style={{ maxHeight: "9in", objectFit: "contain" }} />
            )}
          </div>
        </section>
      </div>
    </main>
  );
}