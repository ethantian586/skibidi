/**
 * Vercel serverless proxy — POST /analyze
 * Buffers the incoming multipart upload then forwards to Modal with the secret key.
 */

const MODAL_URL = "https://ethantian586--sprint-analyzer-web.modal.run";

export const config = {
  api: {
    bodyParser: false,
    responseLimit: "550mb",
    externalResolver: true,
  },
};

// Helper — collect the raw request body into a Buffer
function getRawBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end",  () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ detail: "Method not allowed" });
  }

  try {
    const rawBody = await getRawBody(req);

    const response = await fetch(`${MODAL_URL}/analyze`, {
      method: "POST",
      headers: {
        "X-API-Key":    process.env.API_SECRET,
        "Content-Type": req.headers["content-type"],  // preserves multipart boundary
      },
      body: rawBody,
    });

    // Forward whatever Modal responds — success or error
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      const data = await response.json();
      return res.status(response.status).json(data);
    } else {
      const text = await response.text();
      return res.status(response.status).json({ detail: text });
    }

  } catch (err) {
    console.error("Proxy /analyze error:", err);
    return res.status(502).json({ detail: "Failed to reach processing server." });
  }
}