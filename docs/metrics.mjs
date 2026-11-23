import { drawTraining } from "./charts.mjs";

const formatNumber = value => Number(value).toLocaleString("en-US");
const percent = value => (value * 100).toFixed(1) + "%";
const LABELS = ["Positive", "Negative", "Neutral"];

function matrix(metrics) {
  const values = metrics.confusion.map((row, index) => {
    const total = row.reduce((sum, count) => sum + count, 0);
    return '<div role="rowheader" class="axis">' + LABELS[index] + '</div>' + row.map((count, column) =>
      '<div class="matrix-cell" role="cell" aria-label="' + formatNumber(count) + ' predicted ' + LABELS[column] +
      '" style="--heat:' + (100 * count / (total || 1)).toFixed(1) + '%">' + formatNumber(count) + '</div>'
    ).join("");
  });
  return '<div class="matrix" role="table" aria-label="Confusion matrix">' +
    '<div role="row" class="matrix-row"><div role="columnheader" class="axis">Actual</div>' +
    LABELS.map(name => '<div role="columnheader" class="axis">' + name + '</div>').join("") + '</div>' +
    values.map(row => '<div role="row" class="matrix-row">' + row + '</div>').join("") + '</div>';
}

function createTable(headers, rows) {
  return '<div class="table-scroll" tabindex="0" role="region" aria-label="' + headers[0] + ' data table"><table><thead><tr>' +
    headers.map(label => '<th scope="col">' + label + '</th>').join("") +
    '</tr></thead><tbody>' +
    rows.map(row => '<tr>' + row.map(cell => '<td>' + cell + '</td>').join("") + '</tr>').join("") +
    '</tbody></table></div>';
}

export function renderMetrics(element, metrics, maxTokens) {
  const cards = [
    ["LSTM test accuracy", percent(metrics.testAccuracy), "Correct on held-out data"],
    ["Macro F1", percent(metrics.macroF1), "Each class weighted equally"],
    ["Held-out comments", formatNumber(metrics.samples.test), "Used to evaluate the model"],
    ["Majority baseline", percent(metrics.majorityBaseline), "Always predicts the largest class"],
  ];
  const report = LABELS.map(label => {
    const data = metrics.classReport[label];
    return [label, percent(data.precision), percent(data.recall), percent(data["f1-score"]), formatNumber(data.support)];
  });
  const sources = ["Twitter", "Reddit"].map(source => {
    const data = metrics.sources[source];
    return [source, formatNumber(data.raw), formatNumber(data.clean), formatNumber(data.test), percent(data.testAccuracy)];
  });
  const historyRows = metrics.history.map(row => [row.epoch, percent(row.trainAccuracy), percent(row.validationAccuracy),
    row.trainLoss.toFixed(4), row.validationLoss.toFixed(4)]);
  element.innerHTML =
    '<dl class="metrics-summary">' + cards.map(([label, value, description]) =>
      '<div><dt>' + label + '</dt><dd>' + value + '</dd><small>' + description + '</small></div>').join("") + '</dl>' +
    '<div class="metrics-grid">' +
      '<section class="panel chart-panel" aria-labelledby="training-title"><div class="section-toolbar"><div>' +
        '<h2 id="training-title">Training progress</h2><p>Best checkpoint at epoch ' + metrics.bestEpoch + '</p></div>' +
        '<div class="segmented" role="group" aria-label="Chart metric"><button type="button" data-metric="Accuracy" aria-pressed="true">Accuracy</button>' +
        '<button type="button" data-metric="Loss" aria-pressed="false">Loss</button></div></div>' +
        '<canvas class="training-chart" aria-hidden="true"></canvas><div class="chart-legend">' +
        '<span><i></i>Training</span><span><i></i>Validation</span><span>By epoch</span></div>' +
        '<details class="chart-details"><summary>View epoch data</summary>' +
          createTable(["Epoch", "Train acc.", "Val. acc.", "Train loss", "Val. loss"], historyRows) +
        '</details></section>' +
      '<section class="panel chart-panel" aria-labelledby="matrix-title"><div class="section-toolbar"><div>' +
        '<h2 id="matrix-title">Confusion matrix</h2><p>Held-out comments by actual and predicted label.</p></div></div>' +
        matrix(metrics) + '<p class="chart-caption">Columns show predicted labels. Color intensity shows the share within each actual class.</p></section>' +
    '</div>' +
    '<section class="panel source-panel" aria-labelledby="class-report"><h2 id="class-report">Performance by sentiment</h2>' +
      createTable(["Sentiment", "Precision", "Recall", "F1", "Test comments"], report) + '</section>' +
    '<section class="panel source-panel" aria-labelledby="source-report"><h2 id="source-report">Dataset and splits</h2>' +
      '<p>' + formatNumber(metrics.samples.train) + ' training · ' + formatNumber(metrics.samples.validation) +
      ' validation · ' + formatNumber(metrics.samples.test) + ' test</p>' +
      createTable(["Source", "Raw rows", "Unique usable", "Test comments", "Test accuracy"], sources) +
      '<p>Text columns were normalized across both files. Blank rows, invalid labels, and duplicate inputs were filtered before splitting.</p>' +
      '<p>These metrics describe the original LSTM. The browser also checks for specific contextual cues, including sarcasm and relief at leaving.</p>' +
      '<p>The LSTM reads up to ' + maxTokens + ' tokens of primarily English social text. Test scores describe these datasets. Sarcasm, slang, or a new domain can change performance.</p></section>';

  const chart = element.querySelector(".training-chart");
  let metric = "Accuracy";
  const paint = () => drawTraining(chart, metrics.history, metric, metrics.bestEpoch);
  for (const button of element.querySelectorAll("[data-metric]")) {
    button.addEventListener("click", () => {
      metric = button.dataset.metric;
      for (const toggle of element.querySelectorAll("[data-metric]")) {
        toggle.setAttribute("aria-pressed", String(toggle === button));
      }
      paint();
    });
  }
  new ResizeObserver(paint).observe(chart);
  return paint;
}
