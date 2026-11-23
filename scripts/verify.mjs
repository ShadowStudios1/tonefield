import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { createClassifier } from "../docs/classifier.mjs";

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
console.log("Static app files, model checksum, and inference verified.");
