/* ═══════════════════════════════════════════════════════════════
   QuickHyre — app.js
   Handles: API communication, navigation, charts, tables, live forecast
   ═══════════════════════════════════════════════════════════════ */

const API = 'http://localhost:8000';

// ── Shared state ───────────────────────────────────────────────
let allStates       = [];
let modelRegistry   = {};   // { state: { best_model, metrics } }
let forecastChart   = null;
let liveChart       = null;
let modelsTableData = [];
let modelsSortCol   = 'mape';
let modelsSortAsc   = true;

// ── Model colour map ────────────────────────────────────────────
const MODEL_COLORS = {
  XGBoost: '#4f7cff',
  ARIMA:   '#22c55e',
  Prophet: '#f97316',
  LSTM:    '#a855f7',
};

// ── Utility helpers ─────────────────────────────────────────────
const fmt  = n => n >= 1_000_000
  ? (n / 1_000_000).toFixed(2) + 'M'
  : n.toLocaleString('en-US', { maximumFractionDigits: 0 });

const fmtFull = n => n.toLocaleString('en-US', { maximumFractionDigits: 0 });

const fmtDate = d => {
  const dt = new Date(d + 'T00:00:00');
  return dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
};

function deltaChip(cur, prev) {
  if (prev == null) return '<span class="delta-flat">—</span>';
  const pct = ((cur - prev) / prev * 100).toFixed(1);
  if (pct > 0)  return `<span class="delta-up">▲ ${pct}%</span>`;
  if (pct < 0)  return `<span class="delta-down">▼ ${Math.abs(pct)}%</span>`;
  return '<span class="delta-flat">—</span>';
}

function modelBadge(name, extraClass = 'model-badge') {
  return `<span class="${extraClass} chip--${name}">${name}</span>`;
}

function badgeSpan(name) {
  return `<span class="badge badge-${name}">${name}</span>`;
}

// ── Chart.js default theme ──────────────────────────────────────
Chart.defaults.color          = '#8b95a9';
Chart.defaults.borderColor    = '#2a3347';
Chart.defaults.font.family    = 'Inter, system-ui, sans-serif';

// ── Navigation ──────────────────────────────────────────────────
const TABS = {
  dashboard: { title: 'Dashboard',     subtitle: 'Overview & all states' },
  forecast:  { title: 'Forecast',      subtitle: 'Pre-computed 8-week forecast per state' },
  models:    { title: 'Models',        subtitle: 'Model performance comparison' },
  live:      { title: 'Live Forecast', subtitle: 'Run inference with your own data' },
};

document.querySelectorAll('.nav-item').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    switchTab(tab);
  });
});

function switchTab(tab) {
  document.querySelectorAll('.nav-item').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.toggle('active', p.id === `tab-${tab}`));
  document.getElementById('pageTitle').textContent    = TABS[tab].title;
  document.getElementById('pageSubtitle').textContent = TABS[tab].subtitle;

  if (tab === 'models' && modelsTableData.length > 0) renderModelsTable();
}

// ── Sidebar toggle ──────────────────────────────────────────────
document.getElementById('sidebarToggle').addEventListener('click', () => {
  document.getElementById('sidebar').classList.toggle('collapsed');
});

// ── API helpers ─────────────────────────────────────────────────
async function apiFetch(path, options = {}) {
  const res = await fetch(API + path, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

// ── Health check ────────────────────────────────────────────────
async function checkHealth() {
  const badge     = document.getElementById('apiStatusBadge');
  const statusTxt = document.getElementById('apiStatusText');
  const dot       = document.querySelector('.health-dot');
  const healthTxt = document.querySelector('.health-text');
  try {
    const data = await apiFetch('/health');
    badge.classList.add('online');
    statusTxt.textContent = 'API Online';
    dot.classList.add('ok');
    healthTxt.textContent = `${data.states_loaded} models`;
    document.getElementById('statModelsValue').textContent = data.states_loaded;
  } catch {
    badge.classList.add('offline');
    statusTxt.textContent = 'API Offline';
    dot.classList.add('error');
    healthTxt.textContent = 'Offline';
  }
}

// ── Bootstrap: load states + registry ──────────────────────────
async function bootstrap() {
  await checkHealth();

  try {
    // Fetch states list
    allStates = await apiFetch('/states');
    document.getElementById('statStatesValue').textContent = allStates.length;

    // Populate state dropdowns
    const opts = allStates.map(s => `<option value="${s}">${s}</option>`).join('');
    document.getElementById('forecastState').innerHTML += opts;
    document.getElementById('liveState').innerHTML += opts;

    // Enable run button when state selected
    document.getElementById('forecastState').addEventListener('change', e => {
      document.getElementById('runForecastBtn').disabled = !e.target.value;
    });

    // Fetch model info for all states (needed for dashboard cards & models table)
    const registryArr = await Promise.all(
      allStates.map(s => apiFetch(`/model-info/${encodeURIComponent(s)}`))
    );
    registryArr.forEach(info => { modelRegistry[info.state] = info; });

    renderDashboardStates();
    buildModelsData();
    renderModelsTable();
    renderWinDistribution();

  } catch (e) {
    console.error('Bootstrap error:', e);
  }
}

// ── Dashboard — state cards ──────────────────────────────────────
function renderDashboardStates(filter = '') {
  const grid    = document.getElementById('statesGrid');
  const visible = allStates.filter(s => s.toLowerCase().includes(filter.toLowerCase()));

  if (visible.length === 0) {
    grid.innerHTML = '<p style="color:var(--text-muted);font-size:.9rem;padding:12px;">No states match.</p>';
    return;
  }

  grid.innerHTML = visible.map(state => {
    const info  = modelRegistry[state];
    const model = info?.best_model || '—';
    const mape  = info?.mape?.toFixed(2) ?? '—';
    return `
      <div class="state-card" data-state="${state}" onclick="goToForecast('${state}')">
        <div class="state-card-name">${state}</div>
        <span class="state-card-model chip--${model}">${model}</span>
        <div class="state-card-mape">MAPE: ${mape}%</div>
      </div>`;
  }).join('');
}

document.getElementById('stateSearch').addEventListener('input', e => {
  renderDashboardStates(e.target.value);
});

function goToForecast(state) {
  switchTab('forecast');
  document.getElementById('forecastState').value = state;
  document.getElementById('runForecastBtn').disabled = false;
  runForecast(state);
}

// ── Forecast tab ────────────────────────────────────────────────
document.getElementById('runForecastBtn').addEventListener('click', () => {
  const state = document.getElementById('forecastState').value;
  if (state) runForecast(state);
});

async function runForecast(state) {
  const results  = document.getElementById('forecastResults');
  const empty    = document.getElementById('forecastEmpty');
  const loading  = document.getElementById('forecastLoading');

  results.classList.add('hidden');
  empty.classList.add('hidden');
  loading.classList.remove('hidden');

  try {
    const data = await apiFetch(`/forecast/${encodeURIComponent(state)}`);
    loading.classList.add('hidden');
    renderForecastResults(data);
    results.classList.remove('hidden');
  } catch (e) {
    loading.classList.add('hidden');
    empty.classList.remove('hidden');
    empty.querySelector('p').innerHTML = `<strong>Error:</strong> ${e.message}`;
  }
}

function renderForecastResults(data) {
  document.getElementById('resultState').textContent = data.state;
  document.getElementById('resultModelBadge').className = `model-badge chip--${data.model_used}`;
  document.getElementById('resultModelBadge').textContent = data.model_used;
  document.getElementById('resultMape').textContent = data.mape.toFixed(2) + '%';

  const pts    = data.forecast;
  const labels = pts.map(p => fmtDate(p.date));
  const values = pts.map(p => p.forecasted_sales);

  // Chart
  if (forecastChart) forecastChart.destroy();
  const ctx = document.getElementById('forecastChart').getContext('2d');
  forecastChart = new Chart(ctx, buildChartConfig(labels, values, data.model_used));

  // Table
  const tbody = document.getElementById('forecastTableBody');
  tbody.innerHTML = pts.map((p, i) => `
    <tr>
      <td>Week ${i + 1}</td>
      <td>${fmtDate(p.date)}</td>
      <td><strong>${fmt(p.forecasted_sales)}</strong></td>
      <td>${deltaChip(p.forecasted_sales, i > 0 ? pts[i-1].forecasted_sales : null)}</td>
    </tr>`).join('');
}

// ── Models tab ──────────────────────────────────────────────────
function buildModelsData() {
  modelsTableData = allStates.map(state => {
    const info = modelRegistry[state];
    return {
      state,
      best_model: info.best_model,
      mape:       info.mape,
      all_metrics: info.all_metrics,
    };
  });
}

// Filter buttons
document.getElementById('modelFilterGroup').addEventListener('click', e => {
  const btn = e.target.closest('.filter-btn');
  if (!btn) return;
  document.querySelectorAll('#modelFilterGroup .filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderModelsTable(btn.dataset.filter);
});

// Sortable headers
document.getElementById('modelsTable').querySelector('thead').addEventListener('click', e => {
  const th = e.target.closest('th.sortable');
  if (!th) return;
  const col = th.dataset.col;
  if (col === modelsSortCol) modelsSortAsc = !modelsSortAsc;
  else { modelsSortCol = col; modelsSortAsc = true; }
  renderModelsTable();
});

function renderModelsTable(filter = 'all') {
  const data = filter === 'all'
    ? modelsTableData
    : modelsTableData.filter(r => r.best_model === filter);

  const sorted = [...data].sort((a, b) => {
    let va = a[modelsSortCol], vb = b[modelsSortCol];
    if (typeof va === 'string') va = va.toLowerCase(), vb = vb.toLowerCase();
    return modelsSortAsc ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
  });

  const ALL_MODELS = ['ARIMA', 'Prophet', 'XGBoost', 'LSTM'];
  const tbody = document.getElementById('modelsTableBody');

  tbody.innerHTML = sorted.map(r => {
    const metricCells = ALL_MODELS.map(m => {
      const v = r.all_metrics[m];
      const cls = r.best_model === m ? 'style="font-weight:700;color:var(--text-primary)"' : '';
      return `<td ${cls}>${v != null ? v.toFixed(2) + '%' : '—'}</td>`;
    }).join('');
    return `
      <tr>
        <td>${r.state}</td>
        <td>${badgeSpan(r.best_model)}</td>
        <td><strong>${r.mape.toFixed(2)}%</strong></td>
        ${metricCells}
      </tr>`;
  }).join('');
}

function renderWinDistribution() {
  const counts = {};
  modelsTableData.forEach(r => { counts[r.best_model] = (counts[r.best_model] || 0) + 1; });
  const total   = modelsTableData.length;
  const container = document.getElementById('winDistribution');

  container.innerHTML = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([model, count]) => {
      const pct   = (count / total * 100).toFixed(0);
      const color = MODEL_COLORS[model] || '#4f7cff';
      return `
        <div class="win-bar-group">
          <div class="win-bar-label">
            <span>${badgeSpan(model)}</span>
            <span style="color:var(--text-secondary);font-size:.8rem">${count} states (${pct}%)</span>
          </div>
          <div class="win-bar-track">
            <div class="win-bar-fill" style="width:0%;background:${color}" data-width="${pct}%"></div>
          </div>
        </div>`;
    }).join('');

  // Animate bars after paint
  requestAnimationFrame(() => {
    document.querySelectorAll('.win-bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.width;
    });
  });
}

// ── Live forecast tab ───────────────────────────────────────────
const liveStateEl  = document.getElementById('liveState');
const liveDataEl   = document.getElementById('liveData');
const liveCountEl  = document.getElementById('liveDataCount');
const runLiveBtn   = document.getElementById('runLiveBtn');
const liveErrorEl  = document.getElementById('liveError');

// Count values as user types
liveDataEl.addEventListener('input', () => {
  const vals = parseLiveData();
  liveCountEl.textContent = `${vals.length} value${vals.length !== 1 ? 's' : ''} entered`;
  updateLiveBtnState();
});
liveStateEl.addEventListener('change', updateLiveBtnState);

function parseLiveData() {
  return liveDataEl.value
    .split(/[\s,]+/)
    .map(s => parseFloat(s.replace(/,/g, '')))
    .filter(n => !isNaN(n));
}

function updateLiveBtnState() {
  const ok = liveStateEl.value && parseLiveData().length >= 30;
  runLiveBtn.disabled = !ok;
}

// Load sample data button
document.getElementById('loadSampleBtn').addEventListener('click', async () => {
  const state = liveStateEl.value;
  if (!state) {
    showLiveError('Please select a state first, then load sample data.');
    return;
  }
  try {
    const data = await apiFetch(`/forecast/${encodeURIComponent(state)}`);
    // Use the 8 pre-computed forecast values as a small sample hint;
    // for a realistic 30-value input we prepend synthetic history
    const base = data.forecast[0].forecasted_sales;
    const sample = Array.from({ length: 36 }, (_, i) =>
      Math.round(base * (0.92 + 0.08 * Math.sin(i / 4) + (Math.random() - 0.5) * 0.04))
    );
    liveDataEl.value = sample.join(', ');
    liveDataEl.dispatchEvent(new Event('input'));
    hideLiveError();
  } catch (e) {
    showLiveError(e.message);
  }
});

runLiveBtn.addEventListener('click', runLiveForecast);

async function runLiveForecast() {
  hideLiveError();
  const state  = liveStateEl.value;
  const values = parseLiveData();

  if (!state) return showLiveError('Please select a state.');
  if (values.length < 30) return showLiveError('Please enter at least 30 weekly sales values.');

  const emptyEl   = document.getElementById('liveEmpty');
  const loadingEl = document.getElementById('liveLoading');
  const resultEl  = document.getElementById('liveResult');

  emptyEl.classList.add('hidden');
  resultEl.classList.add('hidden');
  loadingEl.classList.remove('hidden');

  try {
    const data = await apiFetch('/forecast/live', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ state, historical_sales: values }),
    });
    loadingEl.classList.add('hidden');
    renderLiveResults(data);
    resultEl.classList.remove('hidden');
  } catch (e) {
    loadingEl.classList.add('hidden');
    emptyEl.classList.remove('hidden');
    showLiveError(`Inference failed: ${e.message}`);
  }
}

function renderLiveResults(data) {
  document.getElementById('liveResultState').textContent = data.state;
  const mb = document.getElementById('liveResultModelBadge');
  mb.className = `model-badge chip--${data.model_used}`;
  mb.textContent = data.model_used;
  document.getElementById('liveResultMape').textContent = data.mape.toFixed(2) + '%';

  const pts    = data.forecast;
  const labels = pts.map(p => fmtDate(p.date));
  const values = pts.map(p => p.forecasted_sales);

  if (liveChart) liveChart.destroy();
  const ctx = document.getElementById('liveChart').getContext('2d');
  liveChart = new Chart(ctx, buildChartConfig(labels, values, data.model_used));

  const tbody = document.getElementById('liveTableBody');
  tbody.innerHTML = pts.map((p, i) => `
    <tr>
      <td>Week ${i + 1}</td>
      <td>${fmtDate(p.date)}</td>
      <td><strong>${fmt(p.forecasted_sales)}</strong></td>
      <td>${deltaChip(p.forecasted_sales, i > 0 ? pts[i-1].forecasted_sales : null)}</td>
    </tr>`).join('');
}

function showLiveError(msg) {
  liveErrorEl.textContent = msg;
  liveErrorEl.classList.remove('hidden');
}
function hideLiveError() {
  liveErrorEl.classList.add('hidden');
}

// ── Chart builder ───────────────────────────────────────────────
function buildChartConfig(labels, values, model) {
  const color = MODEL_COLORS[model] || '#4f7cff';
  const grad  = null; // will build inside

  return {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label:           'Forecasted Sales',
        data:            values,
        borderColor:     color,
        backgroundColor: hexToRgba(color, 0.12),
        borderWidth:     2.5,
        pointBackgroundColor: color,
        pointRadius:     5,
        pointHoverRadius: 7,
        fill:            true,
        tension:         0.35,
      }],
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: 'index' },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1c2333',
          borderColor:     '#2a3347',
          borderWidth:     1,
          titleColor:      '#e8edf5',
          bodyColor:       '#8b95a9',
          padding:         12,
          callbacks: {
            label: ctx => `  Sales: ${fmtFull(ctx.parsed.y)}`,
          },
        },
      },
      scales: {
        x: {
          grid:  { color: '#2a3347' },
          ticks: { color: '#8b95a9', font: { size: 11 } },
        },
        y: {
          grid:  { color: '#2a3347' },
          ticks: {
            color: '#8b95a9',
            font:  { size: 11 },
            callback: v => fmt(v),
          },
        },
      },
    },
  };
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1,3), 16);
  const g = parseInt(hex.slice(3,5), 16);
  const b = parseInt(hex.slice(5,7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Boot ────────────────────────────────────────────────────────
bootstrap();
