/**
 * Vercel serverless proxy — POST /analyze
 * Forwards the video upload to Modal with the secret API key.
 * The key lives in Vercel's environment variables, never in the browser.
 */

const MODAL_URL = "https://ethantian586--sprint-analyzer-web.modal.run";

export const config = {
  api: {
    bodyParser: false,        // required — we forward raw multipart form data
    responseLimit: "550mb",
  },
};

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ detail: "Method not allowed" });
  }

  try {
    const response = await fetch(`${MODAL_URL}/analyze`, {
      method: "POST",
      headers: {
        "X-API-Key": process.env.API_SECRET,   // injected by Vercel, never visible
        "Content-Type": req.headers["content-type"],
      },
      body: req,                               // stream the raw body straight through
      duplex: "half",                          // required for Node 18+ streaming
    });

    const data = await response.json();
    return res.status(response.status).json(data);

  } catch (err) {
    console.error("Proxy error:", err);
    return res.status(502).json({ detail: "Failed to reach processing server." });
  }
}
