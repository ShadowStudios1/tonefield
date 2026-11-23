import { drawField } from "./charts.mjs";
import { parseCSV, csvValue } from "./csv.mjs";
import { renderMetrics } from "./metrics.mjs";

const $ = id => document.getElementById(id);
const LABELS = ["Positive", "Neutral", "Negative"];
const ROUTES = { playground: "Playground", batch: "Batch analysis", performance: "Model insights", history: "History" };
const EXAMPLES = {
  positive: "The setup was simple and everything worked beautifully. I’d happily recommend it.",
  neutral: "The meeting is in room 204.",
  negative: "This was a complete waste of money. It stopped working the next day.",
};
const MAX_BATCH = 250;
const MAX_TEXT = 5000;
const SAVE_LIMIT = 100;
const STORAGE = { history: "tonefield.history.v1", enabled: "tonefield.history.enabled", theme: "tonefield.theme" };
const formatNumber = value => Number(value).toLocaleString("en-US");
const percent = value => (value * 100).toFixed(1) + "%";
const scoreText = value => value > .999 && value < 1 ? ">99.9%" : value > 0 && value < .001 ? "<0.1%" : percent(value);
const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, char =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
const icon = name => '<svg aria-hidden="true"><use href="./assets/icons.svg#' + name + '"></use></svg>';

function readStorage(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function writeStorage(key, data) {
  try { localStorage.setItem(key, data); return true; } catch { return false; }
}
function loadHistory() {
  try {
    const data = JSON.parse(readStorage(STORAGE.history) || "[]");
    return Array.isArray(data) ? data.filter(row =>
      typeof row.text === "string" && LABELS.includes(row.result?.label) && row.result.scores
    ).slice(0, SAVE_LIMIT) : [];
  } catch { return []; }
}

const state = {
  ready: false, worker: null, processing: false, batchProcessing: false,
  sequence: 0, pending: new Map(), last: null, batch: [], file: null, importSequence: 0,
  saved: loadHistory(), view: null, drawMetrics: null, initialExample: true,
};

function showRoute() {
  const anchor = location.hash.slice(1);
  if (anchor === "main") return;
  const view = Object.hasOwn(ROUTES, anchor) ? anchor : "playground";
  if (state.view === view) return;
  const previous = state.view;
  state.view = view;
  document.querySelectorAll("[data-view]").forEach(section => { section.hidden = section.dataset.view !== view; });
  document.querySelectorAll("[data-nav]").forEach(link => {
    if (link.dataset.nav === view) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  $("breadcrumb-page").textContent = ROUTES[view];
  document.title = ROUTES[view] + " · Tonefield";
  if (view === "history") renderHistory();
  if (previous) {
    window.scrollTo(0, 0);
    $("main").focus({ preventScroll: true });
  }
}

let toastTimeout;
function toast(message) {
  window.clearTimeout(toastTimeout);
  $("toast").textContent = message;
  $("toast").hidden = false;
  toastTimeout = window.setTimeout(() => { $("toast").hidden = true; }, 3000);
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const label = theme === "dark" ? "Switch to light theme" : "Switch to dark theme";
  $("theme-toggle").setAttribute("aria-label", label);
  $("theme-toggle").title = label;
  $("theme-toggle").innerHTML = icon(theme === "dark" ? "sun" : "moon");
  document.querySelector('meta[name="theme-color"]').content = theme === "dark" ? "#121c17" : "#f2f5f1";
  state.drawMetrics?.();
}

function setError(id, message = "") {
  $(id).textContent = message;
  $(id).hidden = !message;
}

function setModelStatus(kind, text) {
  const button = $("model-status");
  button.dataset.state = kind;
  button.firstElementChild.nextElementSibling.textContent = text;
  button.disabled = kind !== "error";
}

function startModel() {
  state.worker?.terminate();
  state.ready = false;
  setModelStatus("loading", "Loading model");
  $("metrics-status").textContent = "Loading evaluation data…";
  $("metrics-status").hidden = false;
  $("metrics-content").hidden = true;
  if (!state.last) renderEmpty("Loading the trained model…");
  syncButtons();

  function failed(message) {
    state.ready = false;
    for (const job of state.pending.values()) job.reject(new Error(message));
    state.pending.clear();
    setModelStatus("error", "Retry model");
    $("metrics-status").textContent = message;
    $("metrics-status").hidden = false;
    if (!state.last) renderEmpty(message);
    syncButtons();
  }

  try {
    const worker = new Worker(new URL("./worker.mjs", import.meta.url), { type: "module" });
    state.worker = worker;
    worker.addEventListener("error", () => failed("The model could not start. Use Retry model to reload it."));
    worker.addEventListener("message", event => {
      if (worker !== state.worker) return;
      const data = event.data;
      if (data.type === "ready") {
        state.ready = true;
        setModelStatus("ready", "Model ready");
        $("strip-count").textContent = formatNumber(data.metrics.samples.total) + " comments";
        $("strip-accuracy").textContent = percent(data.metrics.testAccuracy);
        $("metrics-status").hidden = true;
        $("metrics-content").hidden = false;
        state.drawMetrics = renderMetrics($("metrics-content"), data.metrics, data.maxTokens);
        syncButtons();
        if (state.initialExample && $("single-text").value === EXAMPLES.positive) void analyzeSingle(false);
        else if (!state.last) renderEmpty("Analyze a comment to see its sentiment scores.");
      } else if (data.type === "unavailable") {
        failed(data.message + " Use Retry model to try again.");
      } else if (state.pending.has(data.id)) {
        const job = state.pending.get(data.id);
        if (data.type === "progress") {
          job.progress?.(data.complete, data.total);
        } else {
          state.pending.delete(data.id);
          if (data.type === "results") job.resolve(data.results);
          else job.reject(new Error(data.message));
        }
      }
    });
  } catch {
    failed("This browser could not start an analysis worker. Use Retry model to try again.");
  }
}

function requestAnalysis(texts, progress) {
  return new Promise((resolve, reject) => {
    if (!state.ready) {
      reject(new Error("The model is currently unavailable."));
      return;
    }
    const id = ++state.sequence;
    state.pending.set(id, { resolve, reject, progress });
    state.worker.postMessage({ id, texts });
  });
}

function tag(label, text = label) {
  if (!LABELS.includes(label)) return escapeHtml(text);
  return '<span class="tag ' + label.toLowerCase() + '"><i></i>' + escapeHtml(text) + '</span>';
}

function renderEmpty(message) {
  state.last = null;
  $("copy-result").disabled = true;
  $("single-result").innerHTML = '<div class="output-empty">' + icon("waveform") +
    "<p>" + escapeHtml(message) + "</p></div>";
}

function renderSingle(result, text, source = "") {
  state.last = { result, text };
  $("copy-result").disabled = false;
  const notes = [];
  if (result.totalTokens > result.tokens) notes.push("Only the first " + result.tokens + " tokens were analyzed.");
  if (result.known / result.tokens < .6) notes.push("Many words were outside the model’s vocabulary.");
  $("single-result").innerHTML =
    '<div class="verdict ' + result.label.toLowerCase() + '"><div><small>Predicted sentiment</small>' +
    '<strong class="result-badge"><i></i>' + result.label + '</strong></div>' +
    '<div class="score-big"><small>Model confidence</small><strong>' + escapeHtml(scoreText(result.confidence)) + '</strong></div></div>' +
    '<dl class="scores">' + LABELS.map(label => {
      const score = percent(result.scores[label]);
      return '<div class="score-row ' + label.toLowerCase() + '"><dt>' + label + '</dt>' +
        '<dd class="score-track" aria-hidden="true"><span style="width:' + score + '"></span></dd>' +
        '<dd class="score-number">' + escapeHtml(scoreText(result.scores[label])) + '</dd></div>';
    }).join("") + '</dl>' +
    (notes.length ? '<p class="result-note">' + notes.join(" ") + '</p>' : "") +
    '<div class="result-meta"><span>' + icon("text-aa") + result.tokens + ' tokens</span>' +
    '<span>' + icon("crosshair") + percent(result.known / result.tokens) + ' vocabulary coverage</span>' +
    '<span>' + icon("timer") + Math.max(.1, result.ms).toFixed(1) + ' ms</span>' +
    (source ? '<span>' + escapeHtml(source) + '</span>' : '') + '</div>';
}

function updateSingleInput() {
  const length = $("single-text").value.length;
  $("text-count").textContent = formatNumber(length) + " / 5,000 characters";
  $("clear-single").disabled = !length || state.processing;
  syncButtons();
}

function syncButtons() {
  $("analyze").disabled = !state.ready || state.processing || !$("single-text").value.trim();
  $("analyze-batch").disabled = !state.ready || state.batchProcessing || batchTexts().length === 0;
}

async function analyzeSingle(record = true) {
  if (!state.ready || state.processing) return;
  const input = $("single-text");
  const text = input.value.trim();
  if (!text) {
    setError("single-error", "Enter a comment first.");
    input.focus();
    return;
  }
  state.initialExample = false;
  state.processing = true;
  setError("single-error");
  updateSingleInput();
  const label = $("analyze").querySelector("span");
  label.textContent = "Analyzing…";
  try {
    const [result] = await requestAnalysis([text]);
    if (result.error) {
      setError("single-error", result.error);
      return;
    }
    if (input.value.trim() === text) renderSingle(result, text, record ? "" : "Example result");
    if (record && $("history-enabled").checked) addHistory(text, result);
  } catch (error) {
    setError("single-error", error.message);
  } finally {
    state.processing = false;
    label.textContent = "Analyze sentiment";
    updateSingleInput();
  }
}

function batchTexts() {
  if (state.file?.rows) return state.file.rows
    .map(row => (row[Number($("csv-column").value)] || "").trim()).filter(Boolean);
  return $("batch-text").value.split(/\r?\n|\r/).map(line => line.trim()).filter(Boolean);
}

function setBatchInput() {
  const count = batchTexts().length;
  $("batch-count").textContent = formatNumber(count) + (count === 1 ? " comment" : " comments") +
    (count > MAX_BATCH ? " · limit " + MAX_BATCH : "");
  syncButtons();
}

function clearFile() {
  state.importSequence += 1;
  state.file = null;
  $("batch-file").value = "";
  $("file-label").textContent = "Choose a file or drop it here";
  $("column-picker").hidden = true;
  $("batch-text").readOnly = false;
  $("batch-preview-note").hidden = true;
}

function showFileColumn() {
  const comments = batchTexts();
  $("batch-text").value = comments.slice(0, MAX_BATCH).map(line => line.replace(/[\r\n]+/g, " ")).join("\n").slice(0, 300000);
  setBatchInput();
}

async function importFile(file) {
  if (!file || state.batchProcessing) return;
  clearFile();
  const request = state.importSequence;
  setError("batch-error");
  if (file.size > 1048576 || !/\.(csv|txt)$/i.test(file.name)) {
    setError("batch-error", "Choose a CSV or TXT file that is 1 MB or smaller.");
    return;
  }
  try {
    const text = (await file.text()).replace(/^\uFEFF/, "");
    if (request !== state.importSequence) return;
    if (/\.csv$/i.test(file.name)) {
      const [headers, ...rows] = parseCSV(text);
      if (!headers || !rows.length) throw new Error("The CSV needs a header row and at least one comment.");
      const select = $("csv-column");
      select.replaceChildren(...headers.map((header, index) =>
        new Option(header.trim() || "Column " + (index + 1), index)
      ));
      const preferred = headers.findIndex(header => /^(text|comment|review|content|clean_text|clean_comment|message)$/i.test(header.trim()));
      select.selectedIndex = Math.max(0, preferred);
      state.file = { rows };
      $("column-picker").hidden = false;
      $("batch-text").readOnly = true;
      $("batch-preview-note").hidden = false;
      showFileColumn();
    } else {
      if (text.length > 300000) throw new Error("The TXT exceeds 300,000 characters. Split it into smaller files.");
      $("batch-text").value = text;
      setBatchInput();
    }
    $("file-label").textContent = file.name;
    if (!batchTexts().length) setError("batch-error", "No text was found. Choose another column or file.");
  } catch (error) {
    if (request === state.importSequence) setError("batch-error", error.message);
  }
}

async function analyzeBatch() {
  if (!state.ready || state.batchProcessing) return;
  const texts = batchTexts();
  if (!texts.length || texts.length > MAX_BATCH) {
    setError("batch-error", "Enter between 1 and " + MAX_BATCH + " comments. This input has " + texts.length + ".");
    return;
  }
  const long = texts.findIndex(text => text.length > MAX_TEXT);
  if (long >= 0) {
    setError("batch-error", "Comment " + (long + 1) + " exceeds the 5,000-character limit.");
    return;
  }
  state.batchProcessing = true;
  syncButtons();
  setError("batch-error");
  $("batch-progress").textContent = "Analyzing " + texts.length + " comments…";
  try {
    const results = await requestAnalysis(texts, (complete, total) => {
      $("batch-progress").textContent = complete + " of " + total + " analyzed";
    });
    state.batch = results.map((result, index) => ({ text: texts[index], result, index: index + 1 }));
    $("batch-filter").value = "";
    renderBatch();
    $("batch-progress").textContent = "Batch complete";
  } catch (error) {
    $("batch-progress").textContent = "";
    setError("batch-error", error.message);
  } finally {
    state.batchProcessing = false;
    syncButtons();
  }
}

function resultTable(columns, rows, name) {
  return '<div class="table-scroll" role="region" aria-label="' + name + '" tabindex="0"><table><thead><tr>' +
    columns.map(column => '<th scope="col">' + column + '</th>').join("") +
    '</tr></thead><tbody>' + rows.join("") + '</tbody></table></div>';
}

function renderBatch() {
  const results = state.batch;
  const filter = $("batch-filter").value;
  const valid = results.filter(row => !row.result.error);
  const visible = filter ? valid.filter(row => row.result.label === filter) : results;
  $("batch-actions").hidden = results.length === 0;
  $("batch-distribution").hidden = valid.length === 0;
  $("batch-distribution").innerHTML = LABELS.map(label => tag(label,
    label + " " + valid.filter(row => row.result.label === label).length)).join("");
  $("batch-summary").textContent = results.length
    ? valid.length + " scored" + (results.length > valid.length ? " · " + (results.length - valid.length) + " unscored" : "")
      + (filter ? " · " + visible.length + " shown" : "")
    : "Your predictions will appear here.";
  const rows = visible.map(({ index, text, result }) =>
    '<tr><td class="number-cell">' + index + '</td><td class="table-comment">' + escapeHtml(text) + '</td>' +
    '<td>' + (result.error ? 'Unscored' : tag(result.label)) + '</td><td class="number-cell">' +
    (result.error ? '—' : escapeHtml(scoreText(result.confidence))) + '</td></tr>');
  $("batch-table").innerHTML = rows.length ? resultTable(["#", "Comment", "Sentiment", "Confidence"], rows, "Batch predictions")
    : (results.length ? '<p class="table-empty">No predictions match this filter.</p>' : '');
}

function addHistory(text, result) {
  state.saved.unshift({ text, result, id: crypto.randomUUID(), createdAt: new Date().toISOString() });
  state.saved.length = Math.min(state.saved.length, SAVE_LIMIT);
  if (!writeStorage(STORAGE.history, JSON.stringify(state.saved))) toast("Browser storage is unavailable. History is kept for this tab.");
  renderHistory();
}

function renderHistory() {
  const count = state.saved.length;
  const query = $("history-search").value.trim().toLowerCase();
  const filter = $("history-filter").value;
  const selected = state.saved.filter(row => row.text.toLowerCase().includes(query) && (!filter || row.result.label === filter));
  $("history-count").hidden = !count;
  $("history-count").textContent = String(count);
  $("history-actions").hidden = !count;
  $("history-filters").hidden = !count;
  $("history-summary").textContent = count ? count + (count === 1 ? " prediction saved" : " predictions saved") : "No predictions saved yet.";
  if (!selected.length) {
    $("history-table").innerHTML = '<p class="table-empty">' + (count
      ? "No predictions match your search."
      : 'Run an analysis in the <a class="text-button" href="#playground">playground</a> to build your history.') + '</p>';
    return;
  }
  const rows = selected.map(({ text, result, id, createdAt }) => {
    const date = new Date(createdAt);
    return '<tr><td class="table-comment">' + escapeHtml(text) + '</td><td>' + tag(result.label) +
      '</td><td class="number-cell">' + escapeHtml(scoreText(result.confidence)) +
      '</td><td class="number-cell"><time datetime="' + escapeHtml(createdAt) + '">' +
      escapeHtml(date.toLocaleDateString(undefined, { month: "short", day: "numeric" })) +
      '</time></td><td><button type="button" class="text-button" data-history-id="' +
      escapeHtml(id) + '">Open</button></td></tr>';
  });
  $("history-table").innerHTML = resultTable(["Comment", "Sentiment", "Confidence", "Date", ""], rows, "Saved predictions");
}

function exportRows(rows, name) {
  const records = [["text", "sentiment", "confidence", "positive", "neutral", "negative", "analyzed_at", "error"]];
  for (const { text, result, createdAt } of rows) {
    records.push([text, result.label, result.confidence?.toFixed(6), ...LABELS.map(label => result.scores?.[label].toFixed(6)), createdAt, result.error]);
  }
  const csv = records.map(row => row.map(csvValue).join(",")).join("\r\n");
  const download = document.createElement("a");
  download.href = URL.createObjectURL(new Blob(["\uFEFF", csv], { type: "text/csv;charset=utf-8" }));
  download.download = "tonefield-" + name + ".csv";
  document.body.append(download);
  download.click();
  download.remove();
  window.setTimeout(() => URL.revokeObjectURL(download.href), 1000);
}

const savedTheme = readStorage(STORAGE.theme);
setTheme(savedTheme === "light" || savedTheme === "dark" ? savedTheme : "light");
$("theme-toggle").addEventListener("click", () => {
  const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  setTheme(theme);
  writeStorage(STORAGE.theme, theme);
});
$("shortcut").textContent = /Mac/i.test(navigator.platform) ? "⌘ ↵" : "Ctrl ↵";
document.querySelector(".skip-link").addEventListener("click", event => {
  event.preventDefault();
  $("main").focus();
});
window.addEventListener("hashchange", showRoute);
new ResizeObserver(() => drawField($("field"))).observe($("field"));
$("model-status").addEventListener("click", startModel);

$("single-text").value = EXAMPLES.positive;
$("single-text").addEventListener("input", () => {
  state.initialExample = false;
  setError("single-error");
  if (state.last) renderEmpty("Analyze the updated text to see new scores.");
  updateSingleInput();
});
$("single-text").addEventListener("keydown", event => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !$("analyze").disabled) {
    event.preventDefault();
    $("single-form").requestSubmit();
  }
});
$("single-form").addEventListener("submit", event => {
  event.preventDefault();
  void analyzeSingle();
});
document.querySelector(".example-buttons").addEventListener("click", event => {
  const button = event.target.closest("[data-example]");
  if (!button) return;
  $("single-text").value = EXAMPLES[button.dataset.example];
  $("single-text").dispatchEvent(new Event("input"));
  $("single-text").focus();
});
$("clear-single").addEventListener("click", () => {
  $("single-text").value = "";
  $("single-text").dispatchEvent(new Event("input"));
  renderEmpty("Analyze a comment to see its sentiment scores.");
  $("single-text").focus();
});
$("copy-result").addEventListener("click", async () => {
  if (!state.last) return;
  try {
    const { result } = state.last;
    await navigator.clipboard.writeText(result.label + " (" + scoreText(result.confidence) + " confidence)\n" +
      LABELS.map(label => label + " " + scoreText(result.scores[label])).join(" · "));
    toast("Result copied");
  } catch {
    toast("Clipboard access is unavailable.");
  }
});

$("batch-form").addEventListener("submit", event => {
  event.preventDefault();
  void analyzeBatch();
});
$("batch-text").addEventListener("input", () => {
  clearFile();
  setError("batch-error");
  setBatchInput();
});
$("batch-example").addEventListener("click", () => {
  clearFile();
  $("batch-text").value = [...Object.values(EXAMPLES), "The system is offline for maintenance until 3 pm.",
    "I love the new design. It’s clear, fast and easy to use.", "The app crashes every time I open it."].join("\n");
  setError("batch-error");
  setBatchInput();
});
$("clear-batch").addEventListener("click", () => {
  clearFile();
  $("batch-text").value = "";
  setError("batch-error");
  setBatchInput();
});
$("batch-file").addEventListener("change", event => { void importFile(event.target.files[0]); });
$("csv-column").addEventListener("change", () => {
  showFileColumn();
  setError("batch-error", batchTexts().length ? "" : "This column has no comments.");
});
const upload = $("upload-label");
upload.addEventListener("dragover", event => {
  event.preventDefault();
  upload.classList.add("dragging");
});
upload.addEventListener("dragleave", () => upload.classList.remove("dragging"));
upload.addEventListener("drop", event => {
  event.preventDefault();
  upload.classList.remove("dragging");
  void importFile(event.dataTransfer.files[0]);
});
$("batch-filter").addEventListener("change", renderBatch);
$("export-batch").addEventListener("click", () => exportRows(state.batch, "batch"));

$("history-enabled").checked = readStorage(STORAGE.enabled) !== "false";
$("history-enabled").addEventListener("change", event => {
  if (!writeStorage(STORAGE.enabled, String(event.target.checked))) toast("This preference is available for this tab only.");
});
$("history-search").addEventListener("input", renderHistory);
$("history-filter").addEventListener("change", renderHistory);
$("export-history").addEventListener("click", () => exportRows(state.saved, "history"));
$("confirm-clear").addEventListener("click", () => {
  $("clear-dialog").returnValue = "";
  $("clear-dialog").showModal();
});
$("clear-dialog").addEventListener("close", event => {
  if (event.target.returnValue !== "clear") return;
  state.saved = [];
  if (!writeStorage(STORAGE.history, "[]")) toast("The browser could not clear persistent history.");
  else toast("History cleared");
  renderHistory();
});
$("history-table").addEventListener("click", event => {
  const button = event.target.closest("[data-history-id]");
  if (!button) return;
  const entry = state.saved.find(row => row.id === button.dataset.historyId);
  if (!entry) return;
  state.initialExample = false;
  $("single-text").value = entry.text;
  setError("single-error");
  updateSingleInput();
  renderSingle(entry.result, entry.text, "Saved result");
  location.hash = "playground";
});

showRoute();
updateSingleInput();
setBatchInput();
renderHistory();
startModel();
