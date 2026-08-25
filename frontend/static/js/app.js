/* ==========================================================================
   STOCK AI - Frontend Application Logic
   ========================================================================== */

'use strict';

// ── State ──────────────────────────────────────────────────────────────────
const state = {
    currentSymbol: 'AAPL',
    currentPeriod: '5M',
    currentSteps: 1,
    autoRefreshTimer: null,
    forecastChart: null,
    lastForecastData: null,
};

// ── AI Thoughts — stream of consciousness during analysis ─────────────────
const THOUGHTS = [
    "Fetching latest price data from market feed…",
    "Scanning recent trading sessions for patterns…",
    "Calculating momentum indicators and volatility windows…",
    "Analyzing price trend over the selected period…",
    "Looking at volume patterns and market microstructure…",
    "Identifying support and resistance zones…",
    "Running pattern recognition across recent sessions…",
    "Normalizing features for the analytical models…",
    "Processing historical data through deep analysis layers…",
    "Evaluating multi-day directional signals…",
    "Cross-validating signals using walk-forward analysis…",
    "Computing confidence intervals and price range estimates…",
    "Combining analytical outputs into a unified forecast…",
    "Calibrating uncertainty bounds from recent volatility…",
    "Finalizing forecast for the selected horizon…",
];

// ── DOM Elements ───────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const ui = {
    searchInput:        $('stock-search-input'),
    searchDropdown:     $('search-dropdown'),
    searchClearBtn:     $('search-clear-btn'),
    quoteSymbol:        $('quote-symbol'),
    quoteName:          $('quote-name'),
    quotePrice:         $('quote-price'),
    quoteChange:        $('quote-change'),
    quoteChangePct:     $('quote-change-pct'),
    marketStatusPill:   $('market-status-pill'),
    marketStatusText:   $('market-status-text'),
    quoteDataLabel:     $('quote-data-label'),
    quoteLastUpdated:   $('quote-last-updated'),
    refreshQuoteBtn:    $('refresh-quote-btn'),
    predictBtn:         $('predict-btn'),
    periodSummary:      $('period-summary'),

    // Thinking overlay
    thinkingOverlay:    $('thinking-overlay'),
    currentThought:     $('current-thought'),
    thinkingFill:       $('thinking-fill'),

    // Forecast output
    forecastPrice:      $('forecast-price'),
    forecastReturn:     $('forecast-return'),
    forecastHorizon:    $('forecast-horizon'),
    agreementBadge:     $('agreement-badge'),
    agreementDesc:      $('agreement-desc'),
    forecastRange:      $('forecast-range'),
    forecastSummaryText:$('forecast-summary-text'),
    exportCsvBtn:       $('export-csv-btn'),
    backtestTable:      $('backtest-table'),
};

// ── Chart ──────────────────────────────────────────────────────────────────
function initChart() {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    state.forecastChart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1e293b',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    titleColor: '#9ca3af',
                    bodyColor: '#fff',
                    padding: 12,
                    callbacks: {
                        label: ctx => {
                            if (ctx.dataset.label === 'Range Band') return null;
                            return ` ${ctx.dataset.label}: $${ctx.parsed.y?.toFixed(2) ?? ''}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.04)', drawTicks: false },
                    ticks: { color: '#6b7280', maxRotation: 0, maxTicksLimit: 10, font: { size: 11 } },
                },
                y: {
                    position: 'right',
                    grid: { color: 'rgba(255,255,255,0.05)', drawTicks: false },
                    ticks: { color: '#6b7280', font: { size: 11 }, callback: v => '$' + v.toFixed(0) },
                }
            },
            elements: { point: { radius: 0, hoverRadius: 5 }, line: { tension: 0.3, borderWidth: 2 } }
        }
    });
}

function renderChart(data) {
    if (!state.forecastChart) return;
    const d = data.chart_data;
    const histLen = d.history_dates.length;
    const labels = [...d.history_dates, ...d.forecast_dates];

    const forecastBase = d.history_prices[histLen - 1]; // connect last history point
    const forecastFull = [forecastBase, ...d.forecast_prices];
    const forecastLabels = [d.history_dates[histLen - 1], ...d.forecast_dates];

    // Range band — only over forecast period
    const rangeData = labels.map(lbl => {
        if (d.forecast_dates.includes(lbl)) return null;
        return null;
    });
    const rangeMins = labels.map(lbl => d.forecast_dates.includes(lbl) ? d.range_mins[d.forecast_dates.indexOf(lbl)] : null);
    const rangeMaxs = labels.map(lbl => d.forecast_dates.includes(lbl) ? d.range_maxs[d.forecast_dates.indexOf(lbl)] : null);

    const historyData = [...d.history_prices, ...Array(d.forecast_dates.length).fill(null)];
    const forecastData = [...Array(histLen - 1).fill(null), forecastBase, ...d.forecast_prices];

    state.forecastChart.data.labels = labels;
    state.forecastChart.data.datasets = [
        {
            label: 'Historical',
            data: historyData,
            borderColor: '#06b6d4',
            backgroundColor: 'transparent',
            borderWidth: 1.8,
            pointRadius: 0,
            tension: 0.3,
            order: 2,
        },
        {
            label: 'AI Forecast',
            data: forecastData,
            borderColor: '#f59e0b',
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            borderDash: [7, 4],
            pointRadius: labels.map((_, i) => i >= histLen ? 5 : 0),
            pointBackgroundColor: '#f59e0b',
            tension: 0.35,
            order: 1,
        },
        {
            label: 'Range Band',
            data: rangeMaxs,
            borderColor: 'transparent',
            backgroundColor: 'rgba(245,158,11,0.12)',
            fill: '+1',
            pointRadius: 0,
            tension: 0,
            order: 3,
        },
        {
            label: 'Range Band',
            data: rangeMins,
            borderColor: 'transparent',
            backgroundColor: 'transparent',
            fill: false,
            pointRadius: 0,
            tension: 0,
            order: 3,
        }
    ];
    state.forecastChart.update('active');
}

// ── Thinking Overlay ───────────────────────────────────────────────────────
let _thoughtTimer = null;
let _thoughtIdx = 0;

function startThinking() {
    ui.thinkingOverlay.style.display = 'flex';
    ui.predictBtn.disabled = true;
    ui.thinkingFill.style.width = '0%';

    // Shuffle thoughts
    const thoughts = [...THOUGHTS].sort(() => Math.random() - 0.5);
    _thoughtIdx = 0;

    function showNext() {
        if (_thoughtIdx < thoughts.length) {
            const pct = ((_thoughtIdx + 1) / thoughts.length * 92).toFixed(0);
            ui.thinkingFill.style.width = pct + '%';
            ui.currentThought.textContent = thoughts[_thoughtIdx];
            ui.currentThought.style.animation = 'none';
            void ui.currentThought.offsetWidth; // reflow
            ui.currentThought.style.animation = '';
            _thoughtIdx++;
            _thoughtTimer = setTimeout(showNext, 1100 + Math.random() * 600);
        }
    }
    showNext();
}

function stopThinking() {
    clearTimeout(_thoughtTimer);
    ui.thinkingFill.style.width = '100%';
    setTimeout(() => {
        ui.thinkingOverlay.style.display = 'none';
        ui.thinkingFill.style.width = '0%';
        ui.predictBtn.disabled = false;
    }, 350);
}

// ── Quote ──────────────────────────────────────────────────────────────────
function updateQuoteUI(q) {
    ui.quoteSymbol.textContent = q.symbol;
    ui.quoteName.textContent = q.company_name;
    ui.quotePrice.textContent = '$' + q.price.toFixed(2);
    ui.quoteChange.textContent = (q.change >= 0 ? '+' : '') + q.change.toFixed(2);
    ui.quoteChangePct.textContent = `(${q.change >= 0 ? '+' : ''}${q.change_percent.toFixed(2)}%)`;
    const wrapper = ui.quoteChange.closest('.price-change-wrapper') || ui.quoteChange.parentElement;
    wrapper.className = 'price-change-wrapper ' + (q.change >= 0 ? 'positive' : 'negative');
    ui.quoteDataLabel.textContent = q.data_label || 'Latest available price';
    ui.quoteLastUpdated.textContent = q.last_updated || '--';

    const open = q.is_market_open;
    ui.marketStatusPill.className = 'market-status-pill ' + (open ? 'open' : 'closed');
    ui.marketStatusText.textContent = q.market_status || (open ? 'Market Open' : 'Market Closed');
}

// ── Forecast Output ────────────────────────────────────────────────────────
function renderForecastResults(data) {
    const f = data.forecast;

    // Price card
    ui.forecastPrice.textContent = '$' + f.projected_price.toFixed(2);

    const retPct = f.expected_return_pct;
    const retText = (retPct >= 0 ? '+' : '') + retPct.toFixed(2) + '%';
    ui.forecastReturn.textContent = retText;
    ui.forecastReturn.className = 'highlight-sub ' + (retPct >= 0 ? '' : 'down');

    ui.forecastHorizon.textContent = f.horizon || 'Next Trading Day';

    // Agreement
    const ag = (f.model_agreement || 'HIGH').toUpperCase();
    ui.agreementBadge.textContent = ag;
    ui.agreementBadge.className = 'agreement-badge ' + (ag === 'HIGH' ? 'high' : ag === 'MODERATE' ? 'moderate' : 'low');
    const agScore = Math.round((f.agreement_score || 0.5) * 100);
    ui.agreementDesc.textContent = `${agScore}% signal alignment across analytical layers`;

    // Range
    ui.forecastRange.textContent = `$${f.forecast_range_min.toFixed(2)}  →  $${f.forecast_range_max.toFixed(2)}`;

    // Summary
    const dir = f.direction === 'UP' ? '📈' : f.direction === 'DOWN' ? '📉' : '📊';
    ui.forecastSummaryText.innerHTML = `${dir} ${f.summary_text || ''}`;
}

function renderBacktest(rows) {
    const tbody = ui.backtestTable.querySelector('tbody');
    if (!rows || rows.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-muted">No replay data available.</td></tr>';
        return;
    }
    tbody.innerHTML = rows.map(r => {
        const up = r.actual_direction === 'UP';
        const dn = r.actual_direction === 'DOWN';
        const cls = up ? 'up' : dn ? 'down' : 'neutral';
        return `
            <tr>
                <td class="fw-bold">${r.date}</td>
                <td class="font-mono">$${r.previous_price.toFixed(2)}</td>
                <td class="font-mono fw-bold">$${r.actual_price.toFixed(2)}</td>
                <td class="font-mono ${up ? 'text-green' : dn ? 'text-red' : ''}">${r.actual_change_pct}</td>
                <td><span class="badge-direction ${cls}">${r.actual_direction}</span></td>
            </tr>
        `;
    }).join('');
}

// ── Export CSV ─────────────────────────────────────────────────────────────
function exportCSV() {
    const d = state.lastForecastData;
    if (!d) return alert('No forecast data to export. Run a forecast first.');
    const rows = [['Date', 'Price', 'Type']];
    d.chart_data.history_dates.forEach((dt, i) => rows.push([dt, d.chart_data.history_prices[i], 'Historical']));
    d.chart_data.forecast_dates.forEach((dt, i) => rows.push([dt, d.chart_data.forecast_prices[i], 'AI Forecast']));
    const csv = rows.map(r => r.join(',')).join('\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = `${state.currentSymbol}_forecast_${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
}

// ── Ticker Search ──────────────────────────────────────────────────────────
const QUICK_STOCKS = [
    { symbol: 'AAPL', name: 'Apple Inc.', exchange: 'NASDAQ' },
    { symbol: 'MSFT', name: 'Microsoft Corporation', exchange: 'NASDAQ' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', exchange: 'NASDAQ' },
    { symbol: 'NVDA', name: 'NVIDIA Corporation', exchange: 'NASDAQ' },
    { symbol: 'AMZN', name: 'Amazon.com Inc.', exchange: 'NASDAQ' },
    { symbol: 'META', name: 'Meta Platforms Inc.', exchange: 'NASDAQ' },
    { symbol: 'TSLA', name: 'Tesla Inc.', exchange: 'NASDAQ' },
    { symbol: 'JPM',  name: 'JPMorgan Chase & Co.', exchange: 'NYSE' },
    { symbol: 'V',    name: 'Visa Inc.', exchange: 'NYSE' },
    { symbol: 'JNJ',  name: 'Johnson & Johnson', exchange: 'NYSE' },
    { symbol: 'NFLX', name: 'Netflix Inc.', exchange: 'NASDAQ' },
    { symbol: 'AMD',  name: 'Advanced Micro Devices', exchange: 'NASDAQ' },
    { symbol: 'INTC', name: 'Intel Corporation', exchange: 'NASDAQ' },
    { symbol: 'BABA', name: 'Alibaba Group', exchange: 'NYSE' },
    { symbol: 'RELIANCE.NS', name: 'Reliance Industries', exchange: 'NSE' },
    { symbol: 'TCS.NS', name: 'Tata Consultancy Services', exchange: 'NSE' },
    { symbol: 'INFY.NS', name: 'Infosys Limited', exchange: 'NSE' },
    { symbol: 'HDFC.NS', name: 'HDFC Bank', exchange: 'NSE' },
    { symbol: 'TATAMOTORS.NS', name: 'Tata Motors', exchange: 'NSE' },
];

let searchDebounce;
function handleSearch(query) {
    query = query.trim().toUpperCase();
    ui.searchClearBtn.style.display = query ? 'block' : 'none';
    if (!query) { ui.searchDropdown.style.display = 'none'; return; }

    const filtered = QUICK_STOCKS.filter(s =>
        s.symbol.includes(query) || s.name.toUpperCase().includes(query)
    ).slice(0, 7);

    if (!filtered.length) { ui.searchDropdown.style.display = 'none'; return; }

    ui.searchDropdown.innerHTML = filtered.map(s => `
        <div class="search-item" data-symbol="${s.symbol}">
            <div style="display:flex;flex-direction:column;gap:2px">
                <span class="search-item-sym">${s.symbol}</span>
                <span class="search-item-name">${s.name}</span>
            </div>
            <span class="search-item-ex">${s.exchange}</span>
        </div>
    `).join('');
    ui.searchDropdown.style.display = 'block';

    ui.searchDropdown.querySelectorAll('.search-item').forEach(el => {
        el.addEventListener('click', () => {
            selectStock(el.dataset.symbol);
            ui.searchInput.value = '';
            ui.searchDropdown.style.display = 'none';
            ui.searchClearBtn.style.display = 'none';
        });
    });
}

// ── Select Stock ───────────────────────────────────────────────────────────
function selectStock(symbol) {
    state.currentSymbol = symbol;
    ui.quoteSymbol.textContent = symbol;

    // Update chips
    document.querySelectorAll('.stock-chip').forEach(c => {
        c.classList.toggle('active', c.dataset.symbol === symbol);
    });

    // Fetch fresh quote immediately
    fetchQuote();
}

// ── Forecast ───────────────────────────────────────────────────────────────
async function runForecast() {
    if (ui.predictBtn.disabled) return;

    startThinking();

    try {
        const res = await fetch('/api/forecast', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol: state.currentSymbol,
                period: state.currentPeriod,
                horizon_steps: state.currentSteps,
            }),
        });

        const data = await res.json();
        stopThinking();

        if (!res.ok || !data.success) {
            alert('Forecast error: ' + (data.error || `HTTP ${res.status}`));
            return;
        }

        state.lastForecastData = data;
        updateQuoteUI(data.quote);
        renderForecastResults(data);
        renderChart(data);
        renderBacktest(data.backtest);

    } catch (err) {
        stopThinking();
        console.error(err);
        alert('Network error: ' + err.message);
    }
}

// ── Quote-only refresh ─────────────────────────────────────────────
function setQuoteLoading(loading) {
    const priceEl = ui.quotePrice;
    const changeEl = ui.quoteChange;
    const pctEl    = ui.quoteChangePct;
    if (loading) {
        priceEl.style.opacity  = '0.35';
        changeEl.style.opacity = '0.35';
        pctEl.style.opacity    = '0.35';
        ui.refreshQuoteBtn.style.opacity = '0.5';
        ui.refreshQuoteBtn.disabled = true;
    } else {
        priceEl.style.opacity  = '1';
        changeEl.style.opacity = '1';
        pctEl.style.opacity    = '1';
        ui.refreshQuoteBtn.style.opacity = '1';
        ui.refreshQuoteBtn.disabled = false;
    }
}

async function fetchQuote() {
    setQuoteLoading(true);
    try {
        const res = await fetch('/api/quote?symbol=' + encodeURIComponent(state.currentSymbol));
        if (!res.ok) return;
        const data = await res.json();
        if (data.success) updateQuoteUI(data.quote);
    } catch (_) {}
    finally { setQuoteLoading(false); }
}

// ── Auto-Refresh ───────────────────────────────────────────────────────────
function startAutoRefresh() {
    stopAutoRefresh();
    state.autoRefreshTimer = setInterval(fetchQuote, 30000);
}
function stopAutoRefresh() {
    if (state.autoRefreshTimer) clearInterval(state.autoRefreshTimer);
    state.autoRefreshTimer = null;
}

// ── Wiring ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {

    initChart();

    // Search
    ui.searchInput.addEventListener('input', e => {
        clearTimeout(searchDebounce);
        searchDebounce = setTimeout(() => handleSearch(e.target.value), 200);
    });
    ui.searchClearBtn.addEventListener('click', () => {
        ui.searchInput.value = '';
        ui.searchDropdown.style.display = 'none';
        ui.searchClearBtn.style.display = 'none';
    });
    document.addEventListener('click', e => {
        if (!e.target.closest('.search-input-wrapper') && !e.target.closest('.search-dropdown')) {
            ui.searchDropdown.style.display = 'none';
        }
    });

    // Stock chips
    document.querySelectorAll('.stock-chip').forEach(btn => {
        btn.addEventListener('click', () => selectStock(btn.dataset.symbol));
    });

    // Period toggle
    document.getElementById('period-toggle').addEventListener('click', e => {
        const btn = e.target.closest('[data-period]');
        if (!btn) return;
        document.querySelectorAll('[data-period]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.currentPeriod = btn.dataset.period;
        const months = state.currentPeriod === '1M' ? '1 month' : '5 months';
        ui.periodSummary.innerHTML = `Using: <strong>${months}</strong> of historical trading data`;
    });

    // Step toggle
    document.getElementById('step-toggle').addEventListener('click', e => {
        const btn = e.target.closest('[data-steps]');
        if (!btn) return;
        document.querySelectorAll('[data-steps]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.currentSteps = parseInt(btn.dataset.steps, 10);
    });

    // Auto-refresh toggle
    document.getElementById('auto-refresh-toggle').addEventListener('change', e => {
        if (e.target.checked) startAutoRefresh(); else stopAutoRefresh();
    });

    // Predict button
    ui.predictBtn.addEventListener('click', runForecast);

    // Refresh quote button
    ui.refreshQuoteBtn.addEventListener('click', fetchQuote);

    // Export button
    ui.exportCsvBtn.addEventListener('click', exportCSV);

    // Initial load
    selectStock('AAPL');
    fetchQuote();
});
