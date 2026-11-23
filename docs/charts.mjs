function setupCanvas(canvas) {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return null;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width, height };
}

export function drawField(canvas) {
  const dimensions = setupCanvas(canvas);
  if (!dimensions) return;
  const { context: ctx, width, height } = dimensions;
  const start = width * (width < 460 ? .38 : .37);
  const span = width - start;

  ctx.fillStyle = "#8bd6ad30";
  for (let x = start; x < width; x += 13) {
    for (let y = 12; y < height; y += 13) ctx.fillRect(x, y, 1, 1);
  }

  function coordinate(t, band) {
    const window = Math.pow(Math.max(0, Math.sin(t * Math.PI * .81)), .9);
    const oscillation = Math.sin(t * 13.4 + band * .085) * (32 + 14 * Math.sin(t * 8 + band * .09));
    const x = start + t * span;
    const y = height * .67 - (height * .39 * t) + (band - 11) * (2.8 + window * 3.7) + oscillation * window;
    return [x, y];
  }

  for (let band = 0; band < 24; band += 1) {
    const ink = ctx.createLinearGradient(start, 0, width, 0);
    ink.addColorStop(0, "#77d29f00");
    ink.addColorStop(.37, band % 6 === 1 ? "#90ddae91" : "#5baa8540");
    ink.addColorStop(.69, band % 6 === 1 ? "#b0f1c9db" : "#66ba8b8a");
    ink.addColorStop(1, band % 6 === 1 ? "#68bf9260" : "#509b743c");
    ctx.strokeStyle = ink;
    ctx.lineWidth = band % 6 === 1 ? 1.2 : .8;
    ctx.beginPath();
    for (let step = 0; step <= 116; step += 1) {
      const [x, y] = coordinate(step / 112, band);
      if (step === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  for (const [step, band] of [[.51, 13], [.69, 7], [.82, 19]]) {
    const [x, y] = coordinate(step, band);
    ctx.beginPath();
    ctx.fillStyle = "#b0ecc52c";
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.fillStyle = "#b1f6cb";
    ctx.arc(x, y, 2.4, 0, Math.PI * 2);
    ctx.fill();
  }
}

export function drawTraining(canvas, history, metric, bestEpoch) {
  const dimensions = setupCanvas(canvas);
  if (!dimensions) return;
  const { context: ctx, width, height } = dimensions;
  const accuracy = metric === "Accuracy";
  const style = getComputedStyle(canvas);
  const colors = ["--muted", "--line", "--accent", "--neu-fill"].map(name => style.getPropertyValue(name).trim());
  const [labelColor, grid, trainingColor, valColor] = colors;
  const padding = { left: 42, right: 13, top: 12, bottom: 25 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  const lines = ["train", "validation"].map(name => history.map(row => row[name + metric]));
  const step = accuracy ? .1 : .2;
  const min = Math.floor(Math.min(...lines.flat()) / step) * step;
  const max = Math.max(min + step, Math.ceil(Math.max(...lines.flat()) / step) * step);
  const ticks = Math.round((max - min) / step);
  const x = index => padding.left + (history.length === 1 ? chartWidth / 2 : index * chartWidth / (history.length - 1));
  const y = value => height - padding.bottom - (value - min) * chartHeight / (max - min);

  ctx.font = "700 11px Manrope, sans-serif";
  ctx.strokeStyle = grid;
  ctx.fillStyle = labelColor;
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let index = 0; index <= ticks; index += 1) {
    const value = min + index * step;
    ctx.beginPath();
    ctx.moveTo(padding.left, y(value));
    ctx.lineTo(width - padding.right, y(value));
    ctx.stroke();
    ctx.fillText(accuracy ? Math.round(value * 100) + "%" : value.toFixed(1), padding.left - 7, y(value));
  }

  ctx.textAlign = "center";
  for (let i = 0; i < history.length; i += 1) {
    if (history.length > 10 && i % 2 && i !== history.length - 1) continue;
    ctx.fillText(history[i].epoch, x(i), height - 10);
  }

  const selected = history.findIndex(row => row.epoch === bestEpoch);
  if (selected >= 0) {
    ctx.strokeStyle = valColor;
    ctx.setLineDash([3, 5]);
    ctx.beginPath();
    ctx.moveTo(x(selected), padding.top);
    ctx.lineTo(x(selected), height - padding.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  lines.forEach((values, index) => {
    const color = [trainingColor, valColor][index];
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2.3;
    ctx.beginPath();
    values.forEach((value, point) => point ? ctx.lineTo(x(point), y(value)) : ctx.moveTo(x(point), y(value)));
    ctx.stroke();
    values.forEach((value, point) => {
      ctx.beginPath();
      ctx.arc(x(point), y(value), 2.9, 0, Math.PI * 2);
      ctx.fill();
    });
  });
}
