"""Train the Tonefield LSTM and export its browser inference artifacts."""
import argparse
import copy
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "docs" / "model"
SOURCES = (("Twitter", "Twitter_Data.csv", "clean_text"), ("Reddit", "Reddit_Data.csv", "clean_comment"))
CLASSES = ("Positive", "Negative", "Neutral")
LABELS = {"1": 0, "-1": 1, "0": 2}
MAX_TOKENS = 100
VOCAB_SIZE = 15000
MODEL_SIZE = (128, 64, 32)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*", flags=re.ASCII)
SEED = 42


def tokenize(text):
    return [token.replace("'", "") for token in TOKEN_PATTERN.findall(text.lower().replace("’", "'"))]


def load_dataset(root):
    examples = {}
    conflicts = set()
    counts = {}
    for source, filename, text_column in SOURCES:
        counts[source] = {"raw": 0, "usable": 0, "clean": 0}
        with (root / filename).open(encoding="utf-8-sig", newline="") as file:
            rows = csv.DictReader(file)
            if not {text_column, "category"}.issubset(rows.fieldnames or ()):
                raise ValueError(f"{filename} must have {text_column} and category columns.")
            for row in rows:
                counts[source]["raw"] += 1
                label = LABELS.get((row["category"] or "").strip())
                tokens = tuple(tokenize(row[text_column] or "")[:MAX_TOKENS])
                if not tokens or label is None:
                    continue
                counts[source]["usable"] += 1
                previous = examples.setdefault(tokens, (label, source))
                if previous[0] != label:
                    conflicts.add(tokens)
    records = []
    for tokens, (label, source) in examples.items():
        if tokens not in conflicts:
            records.append((tokens, label, source))
            counts[source]["clean"] += 1
    return records, counts, len(conflicts)


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        embedding, hidden, dense = MODEL_SIZE
        self.embedding = nn.Embedding(vocab_size, embedding, padding_idx=0)
        self.input_dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(embedding, hidden, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.dense = nn.Linear(hidden, dense)
        self.output = nn.Linear(dense, len(CLASSES))
        with torch.no_grad():
            self.lstm.bias_ih_l0[hidden:2 * hidden].fill_(1.0)
            self.lstm.bias_hh_l0.zero_()

    def forward(self, tokens, lengths):
        embedded = self.input_dropout(self.embedding(tokens))
        sequence, _ = self.lstm(embedded)
        last = sequence[torch.arange(len(tokens), device=tokens.device), lengths - 1]
        return self.output(self.dropout(torch.relu(self.dense(self.dropout(last)))))


def build_vocabulary(records, train):
    frequency = Counter(token for index in train for token in records[index][0])
    return {token: index for index, (token, _) in enumerate(frequency.most_common(VOCAB_SIZE - 2), 2)}


def batch_groups(lengths, indices, batch_size):
    # Group similar sequence lengths so the LSTM sees less padding.
    groups = {}
    for index in indices:
        width = min(MAX_TOKENS, ((int(lengths[index]) + 7) // 8) * 8)
        groups.setdefault(width, []).append(index)
    chunks = []
    for width, group in groups.items():
        for offset in range(0, len(group), batch_size):
            chunks.append((width, np.asarray(group[offset:offset + batch_size])))
    return chunks


def run_epoch(model, tensors, batches, loss_fn, optimizer=None):
    tokens, lengths, labels = tensors
    model.train(optimizer is not None)
    losses = 0.0
    correct = 0
    total = 0
    evaluated = []
    with torch.set_grad_enabled(optimizer is not None):
        for width, indices in batches:
            index = torch.as_tensor(indices, device=tokens.device)
            logits = model(tokens[index, :width], lengths[index])
            target = labels[index]
            loss = loss_fn(logits, target)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            predicted = logits.detach().argmax(dim=1)
            losses += loss.item() * len(indices)
            correct += (predicted == target).sum().item()
            total += len(indices)
            if optimizer is None:
                evaluated.append((indices, predicted.cpu().numpy()))
    return losses / total, correct / total, evaluated


def write_atomic(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def export_model(model, vocab, metrics, output):
    output.mkdir(parents=True, exist_ok=True)
    state = model.cpu().state_dict()
    tensors = {
        "embedding": state["embedding.weight"],
        "inputWeights": state["lstm.weight_ih_l0"],
        "hiddenWeights": state["lstm.weight_hh_l0"],
        "gateBias": state["lstm.bias_ih_l0"] + state["lstm.bias_hh_l0"],
        "denseWeights": state["dense.weight"],
        "denseBias": state["dense.bias"],
        "outputWeights": state["output.weight"],
        "outputBias": state["output.bias"],
    }
    offsets = {}
    payload = bytearray()
    for name, tensor in tensors.items():
        offsets[name] = [len(payload) // 4, tensor.numel()]
        payload.extend(tensor.numpy().astype("<f4", copy=False).tobytes())
    version = hashlib.sha256(payload).hexdigest()[:12]
    manifest = {
        "format": 1, "version": version, "labels": CLASSES, "maxTokens": MAX_TOKENS,
        "embeddingSize": MODEL_SIZE[0], "hiddenSize": MODEL_SIZE[1], "denseSize": MODEL_SIZE[2],
        "vocab": vocab, "weights": offsets, "weightBytes": len(payload), "metrics": metrics,
    }
    write_atomic(output / "weights.bin", payload)
    write_atomic(output / "model.json", json.dumps(manifest, ensure_ascii=True, separators=(",", ":")).encode())
    print(f"Exported model {version} to {output}", flush=True)


def train(args):
    torch.manual_seed(SEED)
    torch.set_num_threads(args.threads)
    random = np.random.default_rng(SEED)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    records, sources, conflicts = load_dataset(args.data)
    labels = np.asarray([row[1] for row in records])
    indices = np.arange(len(records))
    strata = [f"{row[2]}-{row[1]}" for row in records]
    train_val, test = train_test_split(indices, test_size=0.2, stratify=strata, random_state=SEED)
    train_indices, val = train_test_split(
        train_val, test_size=0.2, stratify=np.asarray(strata)[train_val], random_state=SEED,
    )
    vocab = build_vocabulary(records, train_indices)
    lengths = np.asarray([len(row[0]) for row in records])
    array = np.zeros((len(records), MAX_TOKENS), dtype=np.int32)
    for index, (words, _, _) in enumerate(records):
        array[index, :len(words)] = [vocab.get(word, 1) for word in words]
    tensors = tuple(torch.as_tensor(items, device=device) for items in (array, lengths, labels))
    val_batches = batch_groups(lengths, val, args.batch_size)
    print(f"Using {device}: {sources}. Excluded {conflicts:,} conflicting token sequences.", flush=True)
    print(f"Split: {len(train_indices):,} train, {len(val):,} validation, {len(test):,} test.", flush=True)

    model = SentimentLSTM(len(vocab) + 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        batches = batch_groups(lengths, random.permutation(train_indices), args.batch_size)
        random.shuffle(batches)
        train_loss, train_accuracy, _ = run_epoch(model, tensors, batches, loss_fn, optimizer)
        val_loss, val_accuracy, _ = run_epoch(model, tensors, val_batches, loss_fn)
        history.append({
            "epoch": epoch, "trainLoss": train_loss, "validationLoss": val_loss,
            "trainAccuracy": train_accuracy, "validationAccuracy": val_accuracy,
        })
        print(f"Epoch {epoch:2d} | train {train_accuracy:.2%} | val {val_accuracy:.2%}"
              f" | val loss {val_loss:.4f} | {time.monotonic() - start:.1f}s", flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 3:
            break

    model.load_state_dict(best_state)
    test_loss, test_accuracy, evaluated = run_epoch(model, tensors, batch_groups(lengths, test, args.batch_size), loss_fn)
    predictions = np.empty(len(records), dtype=np.int64)
    for batch_indices, predicted in evaluated:
        predictions[batch_indices] = predicted
    report = classification_report(labels[test], predictions[test], labels=range(3),
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    print(classification_report(labels[test], predictions[test], labels=range(3),
                                target_names=CLASSES, zero_division=0), flush=True)
    test_sources = np.asarray([records[index][2] for index in test])
    for source, data in sources.items():
        subset = test[test_sources == source]
        data["test"] = len(subset)
        data["testAccuracy"] = float(accuracy_score(labels[subset], predictions[subset]))
    metrics = {
        "sources": sources, "conflictingSequences": conflicts,
        "samples": {"total": len(records), "train": len(train_indices), "validation": len(val), "test": len(test)},
        "classCounts": dict(zip(CLASSES, np.bincount(labels, minlength=3).tolist())),
        "testAccuracy": test_accuracy, "testLoss": test_loss, "macroF1": report["macro avg"]["f1-score"],
        "classReport": {name: report[name] for name in CLASSES},
        "majorityBaseline": float(np.bincount(labels[test]).max() / len(test)),
        "confusion": confusion_matrix(labels[test], predictions[test], labels=range(3)).tolist(),
        "history": history, "bestEpoch": best_epoch, "seed": SEED,
        "parameters": sum(tensor.numel() for tensor in model.parameters()),
    }
    export_model(model, vocab, metrics, args.output)


def predict_cli(args):
    import sys

    with (args.output / "model.json").open(encoding="utf-8") as file:
        model = json.load(file)
    weights = np.fromfile(args.output / "weights.bin", dtype="<f4")
    model_instance = SentimentLSTM(len(model["vocab"]) + 2).eval()
    state = model_instance.state_dict()
    names = {
        "embedding": "embedding.weight", "inputWeights": "lstm.weight_ih_l0",
        "hiddenWeights": "lstm.weight_hh_l0", "gateBias": "lstm.bias_ih_l0",
        "denseWeights": "dense.weight", "denseBias": "dense.bias",
        "outputWeights": "output.weight", "outputBias": "output.bias",
    }
    for exported, original in names.items():
        start, size = model["weights"][exported]
        state[original] = torch.from_numpy(weights[start:start + size]).reshape_as(state[original])
    state["lstm.bias_hh_l0"].zero_()
    model_instance.load_state_dict(state)
    torch.set_num_threads(1)

    def predict(text):
        words = tokenize(text)[:model["maxTokens"]]
        if not words:
            raise ValueError("Enter a comment containing a word or number.")
        ids = [model["vocab"].get(word, 1) for word in words]
        with torch.inference_mode():
            scores = model_instance(torch.tensor([ids]), torch.tensor([len(ids)])).softmax(1)[0].tolist()
        return dict(zip(model["labels"], scores))

    if args.text is not None:
        print(json.dumps(predict(args.text)))
        return
    for line in iter(lambda: input("Enter a comment (or 'exit'): ") if sys.stdin.isatty() else sys.stdin.readline(), ""):
        if line.strip().lower() == "exit":
            return
        try:
            scores = predict(line)
            label = max(scores, key=scores.get)
            print(f"{label} ({scores[label]:.2%})")
        except ValueError as error:
            print(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"], nargs="?", default="train")
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="Directory containing the two source CSVs")
    parser.add_argument("--output", type=Path, default=MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--text")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error("Epochs, batch size, and threads must be positive.")
    if args.command == "train":
        missing = [name for _, name, _ in SOURCES if not (args.data / name).is_file()]
        if missing:
            parser.error(f"Missing in {args.data}: {', '.join(missing)}. Supply the dataset directory with --data.")
        train(args)
    else:
        predict_cli(args)


if __name__ == "__main__":
    main()
