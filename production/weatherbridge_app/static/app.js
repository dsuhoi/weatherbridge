const state = { runs: [], manifest: null, field: "t2m", tau: 3 };
const byId = (id) => document.getElementById(id);
const formatNumber = (value) => Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });

function setStatus(kind, text) {
  byId("status-dot").className = `status-dot ${kind}`;
  byId("status-text").textContent = text;
}

function setOptions(select, values, selected, label) {
  select.replaceChildren(...values.map((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label(value);
    option.selected = value === selected;
    return option;
  }));
}

function renderTauControl() {
  const control = byId("tau-control");
  control.replaceChildren(...state.manifest.interpolation_hours.map((tau) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${tau}h`;
    button.className = tau === state.tau ? "active" : "";
    button.addEventListener("click", () => { state.tau = tau; renderTauControl(); renderForecast(); });
    return button;
  }));
}

function renderForecast() {
  const manifest = state.manifest;
  const field = manifest.fields[state.field];
  const hour = field.hours[String(state.tau)];
  const image = byId("forecast-map");
  image.src = `/api/v1/runs/${encodeURIComponent(manifest.run_id)}/fields/${encodeURIComponent(state.field)}/${state.tau}.webp`;
  image.onload = () => image.parentElement.classList.add("has-image");
  byId("map-title").textContent = `${state.field} · τ=${state.tau}h`;
  byId("valid-time").textContent = new Date(hour.valid_time).toLocaleString();
  byId("model-name").textContent = manifest.model;
  byId("legend-min").textContent = `${formatNumber(field.vmin)} ${field.unit}`;
  byId("legend-max").textContent = `${formatNumber(field.vmax)} ${field.unit}`;
  byId("legend-ramp").style.background = `linear-gradient(90deg, ${field.palette_stops.map(([position, color]) => `${color} ${position * 100}%`).join(",")})`;
  byId("metric-mean").textContent = `${formatNumber(hour.mean)} ${field.unit}`;
  byId("metric-range").textContent = `${formatNumber(hour.min)}…${formatNumber(hour.max)} ${field.unit}`;
  byId("metric-source").textContent = manifest.source;
  byId("metric-grid").textContent = `${manifest.grid.height}×${manifest.grid.width} · ${manifest.grid.resolution_degrees}°`;
}

async function loadManifest(runId) {
  const url = runId ? `/api/v1/runs/${encodeURIComponent(runId)}` : "/api/v1/runs/latest";
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`forecast request failed (${response.status})`);
  state.manifest = await response.json();
  const fields = Object.keys(state.manifest.fields);
  if (!fields.includes(state.field)) state.field = fields[0];
  if (!state.manifest.interpolation_hours.includes(state.tau)) state.tau = state.manifest.interpolation_hours[0];
  setOptions(byId("field-select"), fields, state.field, (value) => value);
  renderTauControl();
  renderForecast();
  setStatus("live", `Updated ${new Date(state.manifest.generated_at).toLocaleTimeString()}`);
}

async function loadRuns() {
  const response = await fetch("/api/v1/runs", { cache: "no-store" });
  if (!response.ok) throw new Error("run index unavailable");
  state.runs = (await response.json()).runs;
  setOptions(byId("run-select"), state.runs.map((run) => run.run_id), state.manifest?.run_id, (runId) => {
    const run = state.runs.find((item) => item.run_id === runId);
    return `${new Date(run.anchor_start).toLocaleString()} · ${run.source}`;
  });
}

async function refresh() {
  try {
    await loadManifest();
    await loadRuns();
  } catch (error) {
    setStatus("error", "No current forecast");
    console.error(error);
  }
}

byId("field-select").addEventListener("change", (event) => { state.field = event.target.value; renderForecast(); });
byId("run-select").addEventListener("change", async (event) => { await loadManifest(event.target.value); });
byId("refresh").addEventListener("click", refresh);
const events = new EventSource("/api/v1/events");
events.addEventListener("forecast", async (event) => {
  const payload = JSON.parse(event.data);
  if (payload.run_id && payload.run_id !== state.manifest?.run_id) await refresh();
});
events.onerror = () => setStatus("error", "Reconnecting");
refresh();

