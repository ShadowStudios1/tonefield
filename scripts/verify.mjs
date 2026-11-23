import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { createClassifier } from "../docs/classifier.mjs";
import { applyContext } from "../docs/context.mjs";

const web = new URL("../docs/", import.meta.url);
const pathFor = path => new URL(path, web);

readFileSync(pathFor(".nojekyll"));
for (const name of readdirSync(web).filter(name => name.endsWith(".mjs"))) {
  const check = spawnSync(process.execPath, ["--check", fileURLToPath(pathFor(name))], { stdio: "inherit" });
  assert.equal(check.status, 0, name + " must be valid JavaScript");
}

const html = readFileSync(pathFor("index.html"), "utf8");
for (const [, asset] of html.matchAll(/(?:src|href)="\.\/([^"#]+)/g)) {
  assert(readFileSync(pathFor(asset)).length > 0, asset + " must be present");
}

const manifest = JSON.parse(readFileSync(pathFor("model/model.json"), "utf8"));
const weights = readFileSync(pathFor("model/weights.bin"));
assert.equal(weights.byteLength, manifest.weightBytes, "The model weight size must match its manifest");
assert.equal(createHash("sha256").update(weights).digest("hex").slice(0, 12), manifest.version,
  "The model weights must match the manifest checkpoint");
assert.deepEqual(manifest.labels, ["Positive", "Negative", "Neutral"]);

const { train, validation, test, total } = manifest.metrics.samples;
assert.equal(train + validation + test, total, "The data splits must match the dataset size");

const buffer = weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength);
const predict = createClassifier(manifest, buffer);
const result = predict("The package arrived on Monday.");
assert(manifest.labels.includes(result.label), "The exported model must predict a valid label");
const probabilities = Object.values(result.scores);
assert.equal(probabilities.length, manifest.labels.length);
assert(probabilities.every(score => Number.isFinite(score) && score >= 0 && score <= 1));
assert(Math.abs(probabilities.reduce((sum, score) => sum + score, 0) - 1) < 1e-6);
const complaintPatterns = [
  "The cabin was lovely, but I was relieved to leave after a sleepless night.",
  "The vehicle looked nice, but we were glad when we could finally leave.",
  "Perfect timing. The form froze while I was filling it out. I had to start over.",
  "Brilliant, the portal reset and now I'm entering my details for the third time.",
  "Fantastic, the editor deleted all my work.",
  "Great, another hour waiting on hold.",
];
for (const complaint of complaintPatterns) {
  const base = predict(complaint);
  const adjusted = applyContext(complaint, base);
  assert.equal(adjusted.label, "Negative", complaint);
  assert.equal(adjusted.scores, base.scores, "Context checks must preserve the LSTM probabilities");
  if (base.label !== "Negative") {
    assert.equal(adjusted.context.modelLabel, base.label);
    assert.equal(adjusted.confidence, null);
  }
}

const noAdjustment = [
  "The visit made me anxious, but I was relieved when the doctor confirmed everything was fine.",
  "I liked the seats, but I wasn't relieved to leave.",
  "Great, the browser crashed, but it recovered everything when I reopened it.",
  "Excellent, the backup saved my files after the app crashed.",
  "Great, the app didn't freeze and I didn't have to retype my answers.",
  "The speaker starts at 11. The program contains two breaks.",
];
for (const text of noAdjustment) {
  const base = predict(text);
  assert.equal(applyContext(text, base), base, text);
}
console.log("Static app files, model checksum, inference, and context checks verified.");
