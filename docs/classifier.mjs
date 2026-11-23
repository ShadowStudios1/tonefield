export function tokenize(text) {
  return (text.toLowerCase().replaceAll("’", "'").match(/[a-z0-9]+(?:'[a-z0-9]+)*/g) || [])
    .map(word => word.replaceAll("'", ""));
}

const sigmoid = value => 1 / (1 + Math.exp(-value));

function dot(weights, offset, values) {
  let sum = 0;
  for (let index = 0; index < values.length; index += 1) sum += weights[offset + index] * values[index];
  return sum;
}

export function createClassifier(manifest, buffer) {
  if (manifest.format !== 1 || buffer.byteLength !== manifest.weightBytes) {
    throw new Error("The model files do not match. Reload to download the current model.");
  }

  const { vocabulary, embeddingSize, hiddenSize, denseSize, labels, maxTokens } = {
    ...manifest, vocabulary: manifest.vocab,
  };
  const lookup = new Map(Object.entries(vocabulary));
  const floats = new Float32Array(buffer);
  const tensors = Object.fromEntries(Object.entries(manifest.weights)
    .map(([name, [start, count]]) => [name, floats.subarray(start, start + count)]));
  const { embedding, inputWeights, hiddenWeights, gateBias, denseWeights, denseBias, outputWeights, outputBias } = tensors;
  const gates = new Float64Array(4 * hiddenSize);
  const cell = new Float64Array(hiddenSize);
  const hidden = new Float64Array(hiddenSize);
  const dense = new Float64Array(denseSize);
  const logits = new Float64Array(labels.length);

  return function predict(text) {
    const startedAt = performance.now();
    const words = tokenize(text);
    if (words.length === 0) return { error: "Enter a comment containing a word or number." };

    cell.fill(0);
    hidden.fill(0);
    let known = 0;
    const count = Math.min(words.length, maxTokens);
    for (let step = 0; step < count; step += 1) {
      const id = lookup.get(words[step]) || 1;
      if (id !== 1) known += 1;
      const vector = embedding.subarray(id * embeddingSize, (id + 1) * embeddingSize);
      for (let gate = 0; gate < gates.length; gate += 1) {
        gates[gate] = gateBias[gate] + dot(inputWeights, gate * embeddingSize, vector)
          + dot(hiddenWeights, gate * hiddenSize, hidden);
      }
      // PyTorch exports the input, forget, candidate and output gates in this order.
      for (let unit = 0; unit < hiddenSize; unit += 1) {
        const input = sigmoid(gates[unit]);
        const forget = sigmoid(gates[unit + hiddenSize]);
        const candidate = Math.tanh(gates[unit + 2 * hiddenSize]);
        const output = sigmoid(gates[unit + 3 * hiddenSize]);
        cell[unit] = forget * cell[unit] + input * candidate;
        hidden[unit] = output * Math.tanh(cell[unit]);
      }
    }

    for (let unit = 0; unit < denseSize; unit += 1) {
      dense[unit] = Math.max(0, denseBias[unit] + dot(denseWeights, unit * hiddenSize, hidden));
    }
    for (let label = 0; label < labels.length; label += 1) {
      logits[label] = outputBias[label] + dot(outputWeights, label * denseSize, dense);
    }

    const maxLogit = Math.max(...logits);
    const scores = Array.from(logits, value => Math.exp(value - maxLogit));
    const total = scores.reduce((sum, score) => sum + score, 0);
    const score = 1 / total;
    return {
      label: labels[logits.indexOf(maxLogit)],
      confidence: score,
      scores: Object.fromEntries(labels.map((label, index) => [label, scores[index] / total])),
      totalTokens: words.length,
      tokens: count,
      known,
      ms: performance.now() - startedAt,
    };
  };
}
