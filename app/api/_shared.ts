// app/api/_shared.ts
import { NextRequest, NextResponse } from "next/server";

/** ---------- Error helper (consistent across routes) ---------- */
export const ERR = {
  NO_IMAGE: "E_NO_IMAGE",
  BAD_URL: "E_BAD_URL",
  MODEL_JSON: "E_MODEL_JSON",
  SERVER: "E_SERVER",
  RATE_LIMIT: "E_RATE_LIMIT",
  TIMEOUT: "E_TIMEOUT",
  CONFIG: "E_CONFIG",
  TOO_LARGE: "E_TOO_LARGE",
} as const;

export type ErrorCode = (typeof ERR)[keyof typeof ERR];

export function errJson(message: string, code: ErrorCode, status = 400) {
  return NextResponse.json({ error: message, error_code: code }, { status });
}

/** ---------- Rate limiting (per-IP, in-memory) ---------- */
export type RLState = { count: number; resetAt: number };

declare global {
  // eslint-disable-next-line no-var
  var __DETECT_RL__: Map<string, RLState> | undefined;
  // eslint-disable-next-line no-var
  var __ANALYZE_RL__: Map<string, RLState> | undefined;
}

export function clientIp(req: NextRequest): string {
  const xf = req.headers.get("x-forwarded-for") || "";
  const ip = xf.split(",")[0].trim() || req.headers.get("x-real-ip") || "0.0.0.0";
  return ip;
}

export function makeRateLimiter(
  bucket: "detect" | "analyze",
  windowMs: number,
  maxRequests: number
) {
  const map =
    bucket === "detect"
      ? (globalThis.__DETECT_RL__ ?? (globalThis.__DETECT_RL__ = new Map<string, RLState>()))
      : (globalThis.__ANALYZE_RL__ ?? (globalThis.__ANALYZE_RL__ = new Map<string, RLState>()));

  return (req: NextRequest) => {
    const ip = clientIp(req);
    const now = Date.now();
    const s = map.get(ip);
    if (!s || s.resetAt <= now) {
      map.set(ip, { count: 1, resetAt: now + windowMs });
      return { ok: true as const };
    }
    if (s.count >= maxRequests) {
      const retryAfter = Math.max(0, Math.ceil((s.resetAt - now) / 1000));
      const res = NextResponse.json(
        { error: "Too many requests. Please wait a moment and try again.", error_code: ERR.RATE_LIMIT },
        { status: 429 }
      );
      res.headers.set("Retry-After", String(retryAfter));
      return { ok: false as const, res };
    }
    s.count += 1;
    return { ok: true as const };
  };
}

/** ---------- Retry with timeout ---------- */
export async function withRetry<T>(
  fn: () => Promise<T>,
  opts: { tries?: number; timeoutMs?: number; baseDelayMs?: number } = {}
): Promise<T> {
  const tries = opts.tries ?? 3;
  const timeoutMs = opts.timeoutMs ?? 10_000;
  const base = opts.baseDelayMs ?? 300;
  let lastErr: unknown;

  const withTimeout = <U>(p: Promise<U>, ms: number) =>
    new Promise<U>((resolve, reject) => {
      const to = setTimeout(() => reject(new Error("timeout")), ms);
      p.then((v) => { clearTimeout(to); resolve(v); })
       .catch((e) => { clearTimeout(to); reject(e); });
    });

  for (let i = 0; i < tries; i++) {
    try {
      // eslint-disable-next-line no-await-in-loop
      return await withTimeout(fn(), timeoutMs);
    } catch (e) {
      lastErr = e;
      // jittered backoff
      // eslint-disable-next-line no-await-in-loop
      await new Promise((r) => setTimeout(r, base * Math.pow(2, i) + Math.random() * 150));
    }
  }
  throw lastErr;
}

/** ---------- Type guards / helpers ---------- */
export function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null;
}

export function hasSize(o: unknown): o is { size: number; type?: string } {
  return isRecord(o) && typeof (o as { size?: unknown }).size === "number";
}