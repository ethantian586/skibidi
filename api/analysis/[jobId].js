/**
 * Vercel serverless proxy — GET /analysis-data/[jobId]
 * Forwards analysis JSON requests to Modal.
 */

const MODAL_URL = process.env.MODAL_URL ?? "https://ethantian586--sprint-analyzer-web.modal.run";

export default async function handler(req, res) {
  const { jobId } = req.query;

  try {
    const response = await fetch(`${MODAL_URL}/analysis/${jobId}`, {
      headers: { "X-API-Key": process.env.API_SECRET },
    });

    const data = await response.json();
    return res.status(response.status).json(data);

  } catch (err) {
    console.error("Proxy error:", err);
    return res.status(502).json({ detail: "Failed to reach processing server." });
  }
}
