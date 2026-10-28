"use strict";

const REFRESH_MS = 3_000;
const COLORS = ["#70e1bf", "#83b8ff", "#f4bd6c", "#d9a8ff", "#ff8585", "#91d5df", "#e8e485", "#f3a6cf"];
const state = { runs: [], selected: new Set(), snapshots: [], metricKeys: [], metricSelection: [], xAxis: "epoch", loading: false };
const runList = document.getElementById("run-list");
const configList = document.getElementById("config-list");
const metricSelect = document.getElementById("metric-select");
const chartGrid = document.getElementById("chart-grid");
const sourceFilter = document.getElementById("source-filter");
const configFilter = document.getElementById("config-filter");
const connectionStatus = document.getElementById("connection-status");
const chartXAxis = document.getElementById("chart-x-axis");

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

async function getJson(url) {
  const response = await fetch(url, { cache: "no-store", credentials: "same-origin" });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

function labelSource(source) {
  if (typeof source === "string") return source;
  if (source && typeof source === "object") {
    return String(source.name || source.kind || source.id || JSON.stringify(source));
  }
  return "unknown source";
}

function renderRunList() {
  runList.replaceChildren();
  if (!state.runs.length) {
    runList.append(node("div", "empty-state compact", "No run folders match these filters."));
    return;
  }
  for (const run of state.runs) {
    const item = node("label", `run-item${state.selected.has(run.run_id) ? " selected" : ""}`);
    const main = node("span", "run-main");
    const checkbox = node("input", "run-check");
    checkbox.type = "checkbox";
    checkbox.checked = state.selected.has(run.run_id);
    checkbox.setAttribute("aria-label", `Compare run ${run.run_id}`);
    checkbox.addEventListener("change", () => selectRun(run.run_id, checkbox.checked));
    const text = node("span", "run-text");
    const titleLine = node("span", "run-title-line");
    titleLine.append(node("span", "run-title", run.run_id));
    if (run.synthetic) titleLine.append(node("span", "synthetic-tag", "SYNTHETIC"));
    if (run.history_limited || run.dropped_count > 0) {
      const tag = node("span", "window-tag", "TRAIN WINDOW");
      tag.title = run.dropped_count > 0
        ? `${run.dropped_count} training events dropped from the retained tail.`
        : "Training history is limited to the retained window.";
      titleLine.append(tag);
    }
    if (run.evaluation_history_limited || run.evaluation_dropped_count > 0) {
      const tag = node("span", "window-tag evaluation", "EVAL WINDOW");
      tag.title = run.evaluation_dropped_count > 0
        ? `${run.evaluation_dropped_count} evaluation records dropped from the retained journal window.`
        : "Evaluation journal history is limited to the retained window.";
      titleLine.append(tag);
    }
    const detail = run.latest_step === null ? labelSource(run.source) : `${labelSource(run.source)} · step ${run.latest_step}`;
    text.append(titleLine, node("span", "run-source", detail));
    main.append(checkbox, text);
    const status = node("span", `status-badge ${run.status || "unread"}`, statusLabel(run.status));
    status.title = [run.error, run.evaluation_error, run.metadata_error, run.last_event_at].filter(Boolean).join(" · ") || "No event timestamp";
    item.append(main, status);
    runList.append(item);
  }
  document.getElementById("selection-count").textContent = `${state.selected.size} / 8`;
}

function statusLabel(status) {
  return ({ live: "live", stale: "stale", no_data: "no data", error: "error", unread: "unread" })[status] || "unknown";
}

function selectRun(runId, selected) {
  if (selected && state.selected.size >= 8) {
    connectionStatus.textContent = "Select up to eight runs";
    renderRunList();
    return;
  }
  if (selected) state.selected.add(runId);
  else state.selected.delete(runId);
  renderRunList();
  refreshSelected();
}

function renderConfigs() {
  configList.replaceChildren();
  if (!state.snapshots.length) {
    configList.append(node("div", "empty-state compact", "Select a run to inspect its configuration."));
    return;
  }
  for (const run of state.snapshots) {
    const entry = node("details", "config-entry");
    const summaryText = `${run.run_id}${run.metadata_status === "incomplete" ? " · metadata incomplete" : ""}`;
    entry.append(node("summary", "", summaryText));
    const config = node("pre");
    config.textContent = JSON.stringify(run.config || {}, null, 2);
    const provenance = node("pre");
    provenance.textContent = JSON.stringify({ source: run.source, provenance: run.provenance || {} }, null, 2);
    const configLabel = node("div", "config-subhead", "CONFIG");
    const provenanceLabel = node("div", "config-subhead", "SOURCE / PROVENANCE");
    entry.append(configLabel, config, provenanceLabel, provenance);
    configList.append(entry);
  }
}

function updateMetricOptions() {
  const selectedValues = new Set(state.metricSelection);
  state.metricKeys = [...new Set(state.snapshots.flatMap((run) => run.metric_keys || []))].sort();
  metricSelect.replaceChildren();
  for (const key of state.metricKeys) {
    const option = node("option", "", key);
    option.value = key;
    option.selected = selectedValues.has(key);
    metricSelect.append(option);
  }
  state.metricSelection = [...metricSelect.selectedOptions].map((option) => option.value);
  const truncated = state.snapshots.some((run) => run.metric_keys_truncated);
  document.getElementById("metric-count").textContent = `${state.metricKeys.length} metric keys${truncated ? " · key list capped" : ""}`;
}

async function refreshRuns() {
  if (state.loading) return;
  state.loading = true;
  try {
    const query = new URLSearchParams({ source: sourceFilter.value.trim(), config: configFilter.value.trim() });
    const data = await getJson(`/api/runs?${query}`);
    state.runs = data.runs || [];
    connectionStatus.classList.remove("error");
    connectionStatus.textContent = `${state.runs.length} recent runs · refreshed`;
    renderRunList();
    await refreshSelected();
    document.getElementById("updated-at").textContent = `Updated ${new Date().toLocaleTimeString()}`;
  } catch (error) {
    connectionStatus.classList.add("error");
    connectionStatus.textContent = `Connection error · ${error.message}`;
  } finally {
    state.loading = false;
  }
}

async function refreshSelected() {
  const ids = [...state.selected];
  if (!ids.length) {
    state.snapshots = [];
    renderConfigs();
    updateMetricOptions();
    renderCharts();
    document.getElementById("synthetic-banner").classList.add("hidden");
    document.getElementById("history-banner").classList.add("hidden");
    return;
  }
  try {
    const query = new URLSearchParams({ ids: ids.join(","), limit: "1200" });
    const data = await getJson(`/api/run?${query}`);
    state.snapshots = data.runs || [];
    renderConfigs();
    updateMetricOptions();
    renderCharts();
    const synthetic = state.snapshots.some((run) => run.synthetic);
    document.getElementById("synthetic-banner").classList.toggle("hidden", !synthetic);
    const limitedRuns = state.snapshots.filter((run) => run.history_limited || run.evaluation_history_limited);
    const historyBanner = document.getElementById("history-banner");
    historyBanner.classList.toggle("hidden", !limitedRuns.length);
    if (limitedRuns.length) {
      const details = limitedRuns.map((run) => {
        const limits = [];
        if (run.history_limited) limits.push(run.dropped_count ? `train −${run.dropped_count}` : "train windowed");
        if (run.evaluation_history_limited) limits.push(run.evaluation_dropped_count ? `eval −${run.evaluation_dropped_count}` : "eval windowed");
        return `${run.run_id} (${limits.join(", ")})`;
      });
      historyBanner.textContent = `History window limited: ${details.join("; ")}. Charts use retained events; older records may be outside this view.`;
    }
  } catch (error) {
    connectionStatus.classList.add("error");
    connectionStatus.textContent = `Run read error · ${error.message}`;
  }
}

function findKeys(predicate) {
  return state.metricKeys.filter((key) => predicate(key.toLowerCase()));
}

function exactKeys(...keys) {
  const have = new Set(state.metricKeys);
  return keys.filter((key) => have.has(key));
}

function targetScoreKeys(target) {
  const targetPattern = new RegExp(`(?:^|[/_])${target}(?:$|[/_])`, "i");
  return findKeys((key) => (key.startsWith("train_eval/") || key.startsWith("valid/"))
    && (key.includes("score") || key.includes("/wp/")) && targetPattern.test(key));
}

function comparableScoreKeys() {
  return findKeys((key) => (key.startsWith("train_eval/") || key.startsWith("valid/"))
    && (key === "train_eval/score" || key === "valid/score" || key.includes("/wp/")));
}

function metricGroups() {
  return [
    { title: "Comparable loss", subtitle: "train_eval/loss and valid/loss · same evaluation path", keys: exactKeys("train_eval/loss", "valid/loss"), wide: true },
    { title: "Online training loss", subtitle: "train/loss is collected during updates; it is not comparable to evaluation loss", keys: exactKeys("train/loss"), dashed: true },
    { title: "Comparable score / WP", subtitle: "Comparable recorded score or weighted Pearson (WP) series", keys: comparableScoreKeys(), wide: true },
    { title: "Per-target score · t0", subtitle: "Recorded t0 score series for train_eval and valid", keys: targetScoreKeys("t0") },
    { title: "Per-target score · t1", subtitle: "Recorded t1 score series for train_eval and valid", keys: targetScoreKeys("t1") },
    { title: "Learning rate", subtitle: "Recorded optimizer learning rate", keys: findKeys((key) => key === "optimizer/lr" || key.startsWith("optimizer/lr/") || key.endsWith("/learning_rate")) },
    { title: "Gradient norm", subtitle: "Recorded optimizer gradient norm", keys: findKeys((key) => key.startsWith("optimizer/") && key.includes("grad") && key.includes("norm")) },
    { title: "Gradient clipping", subtitle: "Recorded clipping threshold, fraction, or count", keys: findKeys((key) => key.startsWith("optimizer/") && key.includes("clip")) },
    { title: "Branch norms and activation RMS", subtitle: "Recorded local and global norms, activation, hidden, or cell RMS", keys: findKeys((key) => (key.startsWith("local/") || key.startsWith("global/")) && !key.includes("grad") && ["norm", "rms", "activation", "hidden", "cell"].some((token) => key.includes(token))) },
    { title: "Branch gradients", subtitle: "Recorded local and global branch gradients", keys: findKeys((key) => (key.startsWith("local/") || key.startsWith("global/")) && key.includes("grad")) },
    { title: "Branch gates", subtitle: "Recorded local or global gate mean and saturation", keys: findKeys((key) => (key.startsWith("local/gate/") || key.startsWith("global/gate/")) && (key.endsWith("/mean") || key.includes("saturat"))) },
    { title: "Fusion saturation", subtitle: "Recorded fusion gate or saturation diagnostics", keys: findKeys((key) => key.startsWith("fusion/") && (key.includes("gate") || key.includes("saturat"))) },
    { title: "Attention entropy", subtitle: "Recorded attention entropy diagnostics", keys: findKeys((key) => key.startsWith("attention/") && key.includes("entropy")) },
    { title: "Attention lag", subtitle: "Recorded attention lag diagnostics", keys: findKeys((key) => key.startsWith("attention/") && key.includes("lag")) },
    { title: "Relative position encoding / bias", subtitle: "Recorded relative-position or relative-bias metrics", keys: findKeys((key) => key.startsWith("relative_bias/")) },
    { title: "Prediction mean, standard deviation, and clipping", subtitle: "Recorded prediction distribution summaries", keys: findKeys((key) => (key.startsWith("prediction/") || key.startsWith("train/prediction/") || key.includes("/prediction_")) && ["mean", "std", "clip", "fraction"].some((token) => key.includes(token))), wide: true },
    { title: "GPU utilization", subtitle: "Recorded GPU utilization", keys: findKeys((key) => key.startsWith("gpu/") && (key.includes("util") || key.includes("usage"))) },
    { title: "GPU memory", subtitle: "Recorded GPU memory use", keys: findKeys((key) => key.startsWith("gpu/") && (key.includes("memory") || key.includes("mem"))) },
    { title: "GPU power", subtitle: "Recorded GPU power draw", keys: findKeys((key) => key.startsWith("gpu/") && key.includes("power")) },
    { title: "CPU utilization", subtitle: "Recorded CPU utilization", keys: findKeys((key) => key.startsWith("cpu/") && (key.includes("util") || key.includes("usage"))) },
    { title: "CPU memory / RSS", subtitle: "Recorded process or host resident memory", keys: findKeys((key) => key.startsWith("cpu/") && (key.includes("rss") || key.includes("memory"))) },
    { title: "Throughput", subtitle: "Recorded rows, samples, or steps per second", keys: findKeys((key) => key.startsWith("throughput/")) },
    { title: "Phase timing", subtitle: "Recorded training, evaluation, and diagnostic phase times", keys: findKeys((key) => key.startsWith("timing/")), wide: true },
  ];
}

function renderCharts() {
  chartGrid.replaceChildren();
  if (!state.snapshots.length) {
    chartGrid.append(node("div", "empty-state panel", "Select one or more runs to view recorded metrics."));
    return;
  }
  for (const group of metricGroups()) renderChartCard(group);
  if (state.metricSelection.length) {
    renderChartCard({
      title: "Custom metric comparison",
      subtitle: "Selected slash-keyed metrics · markers show recorded observations only",
      keys: state.metricSelection,
      wide: true,
    });
  }
}

function buildSeries(keys, customDashed) {
  const series = [];
  state.snapshots.forEach((run, runIndex) => {
    const events = run.events || [];
    const timeOrigin = events.length ? Date.parse(events[0].wall_time) : Number.NaN;
    for (const key of keys) {
      const points = events.map((event) => {
        const metrics = event.metrics || {};
        const value = metrics[key];
        let x = Number.NaN;
        if (state.xAxis === "epoch") x = event.epoch;
        else if (state.xAxis === "step") x = event.step;
        else {
          const eventTime = Date.parse(event.wall_time);
          if (Number.isFinite(eventTime) && Number.isFinite(timeOrigin)) x = (eventTime - timeOrigin) / 1_000;
        }
        return {
          x: Number.isFinite(x) ? x : null,
          y: typeof value === "number" && Number.isFinite(value) ? value : null,
          epoch: event.epoch,
          step: event.step,
          wallTime: event.wall_time,
        };
      });
      if (!points.some((point) => point.y !== null && point.x !== null)) continue;
      const isValid = key.startsWith("valid/");
      const isOnline = key === "train/loss";
      series.push({
        label: `${run.synthetic ? "SYNTHETIC DEMO · " : ""}${run.run_id} · ${key}`,
        color: COLORS[runIndex % COLORS.length],
        dash: customDashed || isValid || isOnline,
        points,
      });
    }
  });
  return series;
}

function renderChartCard(group) {
  const card = node("article", `chart-card${group.wide ? " wide" : ""}`);
  const header = node("div", "chart-card-header");
  const titleBox = node("div");
  const demoSelected = state.snapshots.some((run) => run.synthetic);
  const historyLimited = state.snapshots.some((run) => run.history_limited || run.evaluation_history_limited);
  const title = `${demoSelected ? "SYNTHETIC DEMO · " : ""}${historyLimited ? "WINDOWED HISTORY · " : ""}${group.title}`;
  titleBox.append(node("h3", "", title), node("p", "chart-subtitle", group.subtitle));
  header.append(titleBox);
  card.append(header);
  const series = buildSeries(group.keys || [], group.dashed);
  if (!series.length) {
    const empty = node("div", "chart-empty", "No measurement recorded for these metric names in the selected runs.");
    card.append(empty);
    chartGrid.append(card);
    return;
  }
  const wrap = node("div", "chart-wrap");
  const canvas = node("canvas", "chart-canvas");
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", `${group.title} chart`);
  const tooltip = node("div", "chart-tooltip hidden");
  wrap.append(canvas, tooltip);
  const legend = node("div", "chart-legend");
  for (const item of series) {
    const entry = node("span", "legend-entry");
    const swatch = node("span", `legend-line${item.dash ? " dashed" : ""}`);
    swatch.style.color = item.color;
    entry.append(swatch, node("span", "", item.label));
    legend.append(entry);
  }
  card.append(wrap, legend);
  chartGrid.append(card);
  drawChart(canvas, series, tooltip);
}

function formatNumber(value) {
  if (Math.abs(value) >= 1_000 || (Math.abs(value) > 0 && Math.abs(value) < .001)) return value.toExponential(2);
  return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function axisLabel() {
  return ({ epoch: "Epoch", step: "Training step", elapsed: "Elapsed wall time (s)" })[state.xAxis];
}

function drawChart(canvas, series, tooltip) {
  const bounds = canvas.getBoundingClientRect();
  const width = Math.max(250, bounds.width);
  const height = Math.max(170, bounds.height);
  const ratio = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  const plot = { left: 48, right: width - 12, top: 12, bottom: height - 31 };
  const measured = series.flatMap((line) => line.points.filter((point) => point.y !== null && point.x !== null));
  const values = measured.map((point) => point.y);
  const xValues = measured.map((point) => point.x);
  let minY = Math.min(...values);
  let maxY = Math.max(...values);
  let minX = Math.min(...xValues);
  let maxX = Math.max(...xValues);
  if (minY === maxY) { minY -= Math.abs(minY) * .05 || 1; maxY += Math.abs(maxY) * .05 || 1; }
  if (minX === maxX) { minX -= .5; maxX += .5; }
  const hits = [];
  context.font = "10px Inter, system-ui, sans-serif";
  context.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick += 1) {
    const y = plot.top + (plot.bottom - plot.top) * tick / 4;
    const value = maxY - (maxY - minY) * tick / 4;
    context.strokeStyle = "#24313d";
    context.beginPath(); context.moveTo(plot.left, y); context.lineTo(plot.right, y); context.stroke();
    context.fillStyle = "#7f91a2";
    context.textAlign = "right";
    context.textBaseline = "middle";
    context.fillText(formatNumber(value), plot.left - 7, y);
  }
  context.fillStyle = "#7f91a2";
  context.textAlign = "left";
  context.textBaseline = "top";
  context.fillText(formatNumber(minX), plot.left, plot.bottom + 10);
  context.textAlign = "right";
  context.fillText(formatNumber(maxX), plot.right, plot.bottom + 10);
  context.textAlign = "center";
  context.textBaseline = "bottom";
  context.fillText(axisLabel(), (plot.left + plot.right) / 2, height - 2);
  for (const line of series) {
    context.strokeStyle = line.color;
    context.fillStyle = line.color;
    context.lineWidth = 1.8;
    context.setLineDash(line.dash ? [5, 4] : []);
    context.beginPath();
    const linePoints = line.points.filter((point) => point.y !== null && point.x !== null);
    for (const [index, point] of linePoints.entries()) {
      const x = plot.left + (point.x - minX) / (maxX - minX) * (plot.right - plot.left);
      const y = plot.bottom - (point.y - minY) / (maxY - minY) * (plot.bottom - plot.top);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
      hits.push({ x, y, line, point });
    }
    context.stroke();
    context.setLineDash([]);
    for (const point of linePoints) {
      const x = plot.left + (point.x - minX) / (maxX - minX) * (plot.right - plot.left);
      const y = plot.bottom - (point.y - minY) / (maxY - minY) * (plot.bottom - plot.top);
      context.beginPath();
      context.arc(x, y, 2.8, 0, Math.PI * 2);
      context.fill();
    }
  }
  canvas._chartHits = hits;
  canvas.onmousemove = (event) => showHover(canvas, tooltip, event);
  canvas.onmouseleave = () => tooltip.classList.add("hidden");
}

function showHover(canvas, tooltip, event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const hits = canvas._chartHits || [];
  let nearest = null;
  let distance = 100;
  for (const hit of hits) {
    const candidate = Math.hypot(hit.x - x, hit.y - y);
    if (candidate < distance) { nearest = hit; distance = candidate; }
  }
  if (!nearest) { tooltip.classList.add("hidden"); return; }
  const measuredX = state.xAxis === "elapsed"
    ? `elapsed ${formatNumber(nearest.point.x)}s`
    : `${axisLabel().toLowerCase()} ${formatNumber(nearest.point.x)}`;
  const time = nearest.point.wallTime ? ` · ${nearest.point.wallTime}` : "";
  tooltip.textContent = `${nearest.line.label} · ${measuredX} · step ${nearest.point.step} · ${formatNumber(nearest.point.y)}${time}`;
  tooltip.style.left = `${Math.min(x + 10, rect.width - 240)}px`;
  tooltip.style.top = `${Math.max(4, y - 34)}px`;
  tooltip.classList.remove("hidden");
}

let filterTimer = null;
for (const input of [sourceFilter, configFilter]) {
  input.addEventListener("input", () => {
    window.clearTimeout(filterTimer);
    filterTimer = window.setTimeout(refreshRuns, 250);
  });
}
metricSelect.addEventListener("change", () => {
  state.metricSelection = [...metricSelect.selectedOptions].map((option) => option.value);
  renderCharts();
});
chartXAxis.addEventListener("change", () => {
  state.xAxis = chartXAxis.value;
  const note = state.xAxis === "elapsed"
    ? "Elapsed time starts at the first retained event in each run."
    : `${axisLabel()} values come directly from each event.`;
  document.getElementById("chart-axis-note").textContent = note;
  renderCharts();
});
document.getElementById("refresh-now").addEventListener("click", refreshRuns);
window.addEventListener("resize", () => renderCharts());
refreshRuns();
window.setInterval(refreshRuns, REFRESH_MS);
