/**
 * Vercel serverless proxy — GET /download/[jobId]
 * Streams the processed video from Modal back to the browser.
 */

const MODAL_URL = "https://ethantian586--sprint-analyzer-web.modal.run";

export const config = {
  api: {
    responseLimit: "550mb",
  },
};

export default async function handler(req, res) {
  const { jobId } = req.query;

  try {
    const response = await fetch(`${MODAL_URL}/download/${jobId}`, {
      headers: { "X-API-Key": process.env.API_SECRET },
    });

    if (!response.ok) {
      const data = await response.json();
      return res.status(response.status).json(data);
    }

    // Stream the video straight through to the browser
    res.setHeader("Content-Type", "video/mp4");
    res.setHeader(
      "Content-Disposition",
      `attachment; filename="sprint_${jobId.slice(0, 8)}.mp4"`
    );

    const reader = response.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
    res.end();

  } catch (err) {
    console.error("Proxy error:", err);
    return res.status(502).json({ detail: "Failed to reach processing server." });
  }
}