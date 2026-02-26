/**
 * Vercel serverless — GET /upload-token
 *
 * Mints a short-lived HMAC-SHA256 upload token so the browser can POST
 * a video file directly to Modal without ever seeing the real API_SECRET.
 *
 * Token format:  "<expires_unix>.<hex_hmac>"
 * Token lifetime: 5 minutes
 */

import { createHmac } from "crypto";

const MODAL_URL   = process.env.MODAL_URL ?? "https://ethantian586--sprint-analyzer-web.modal.run";
const TOKEN_TTL_S = 5 * 60; // 5 minutes

export default function handler(req, res) {
  if (req.method !== "GET") {
    return res.status(405).json({ detail: "Method not allowed" });
  }

  const secret = process.env.API_SECRET;
  if (!secret) {
    return res.status(500).json({ detail: "Server misconfiguration." });
  }

  const expires = Math.floor(Date.now() / 1000) + TOKEN_TTL_S;
  const payload = String(expires);
  const sig     = createHmac("sha256", secret).update(payload).digest("hex");
  const token   = `${payload}.${sig}`;

  // Cache-Control: no-store so the token is never cached by a CDN or browser
  res.setHeader("Cache-Control", "no-store");
  return res.status(200).json({
    upload_url: `${MODAL_URL}/analyze`,
    token,
    expires_at: expires,
  });
}
