// app/api/cache-image/route.ts
import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge"; // fast + CDN caching

// Only cache from these hosts (your 4 samples)
const ALLOW = new Set<string>([
  "preview.redd.it",
  "diminishedvalueofgeorgia.com",
  "i.redd.it",
  "brightonpanelworks.com.au",
]);

export async function GET(req: NextRequest) {
  try {
    const src = req.nextUrl.searchParams.get("url");
    if (!src) return NextResponse.json({ error: "Missing ?url" }, { status: 400 });

    let u: URL;
    try { u = new URL(src); } catch { return NextResponse.json({ error: "Bad url" }, { status: 400 }); }
    if (!/^https?:$/.test(u.protocol)) return NextResponse.json({ error: "Must be http(s)" }, { status: 400 });
    if (!ALLOW.has(u.hostname)) return NextResponse.json({ error: "Host not allowed" }, { status: 403 });

    // Ask Vercel to cache at the CDN; also set strong headers.
    const upstream = await fetch(u.toString(), {
      // Edge runtime supports Next.js fetch caching hints:
      next: { revalidate: 60 * 60 * 24 * 365 }, // 1 year revalidation window
    });

    if (!upstream.ok) {
      return NextResponse.json({ error: `Upstream ${upstream.status}` }, { status: 502 });
    }

    // Stream body through with long-lived, immutable caching for the CDN.
    const res = new NextResponse(upstream.body, {
      status: 200,
      headers: {
        "Content-Type": upstream.headers.get("content-type") || "image/jpeg",
        "Cache-Control": "public, s-maxage=31536000, immutable", // CDN: 1y
        "CDN-Cache-Control": "public, s-maxage=31536000, immutable",
        "Vary": "Accept-Encoding",
      },
    });
    return res;
  } catch {
    return NextResponse.json({ error: "Proxy error" }, { status: 500 });
  }
}
