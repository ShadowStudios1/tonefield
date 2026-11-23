import { createClassifier } from "./classifier.mjs";

let predict;

async function load() {
  const response = await fetch("./model/model.json", { cache: "no-cache" });
  if (!response.ok) throw new Error("The model metadata could not be loaded.");
  const manifest = await response.json();
  const binary = await fetch("./model/weights.bin?v=" + encodeURIComponent(manifest.version));
  if (!binary.ok) throw new Error("The model weights could not be loaded.");
  predict = createClassifier(manifest, await binary.arrayBuffer());
  postMessage({ type: "ready", metrics: manifest.metrics, maxTokens: manifest.maxTokens });
}

self.onmessage = event => {
  const { id, texts } = event.data;
  if (!predict) return;
  try {
    const results = texts.map((text, index) => {
      const result = predict(text);
      if (texts.length > 1 && (index + 1) % 12 === 0) {
        postMessage({ type: "progress", id, complete: index + 1, total: texts.length });
      }
      return result;
    });
    postMessage({ type: "results", id, results });
  } catch {
    postMessage({ type: "failure", id, message: "Analysis failed. Try again with different text." });
  }
};

load().catch(error => postMessage({ type: "unavailable", message: error.message }));
