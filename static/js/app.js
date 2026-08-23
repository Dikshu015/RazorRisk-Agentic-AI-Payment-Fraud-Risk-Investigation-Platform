// Report uncaught frontend errors into the server-side audit trail
// (logs/frontend_client.log) instead of a browser console only the person
// hitting the bug ever sees — best-effort, never throws itself.
function logClientError(message, context) {
    try {
        fetch('/api/v1/logs/client', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ level: 'error', message: String(message).slice(0, 500), context })
        }).catch(() => {});
    } catch (e) { /* logging must never itself break the page */ }
}
window.addEventListener('error', (e) => logClientError(e.message, 'window.onerror'));
window.addEventListener('unhandledrejection', (e) => logClientError(e.reason, 'unhandledrejection'));

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    loadStats();
    loadRecentTransactions();
    loadAgentStatus();
    refreshLogStream();

    // Form submit listener
    const form = document.getElementById('scoring-form');
    if (form) {
        form.addEventListener('submit', handleTransactionScore);
    }
});

// Tab Navigation
function initTabs() {
    const buttons = document.querySelectorAll('.nav-btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            buttons.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));

            btn.classList.add('active');
            const tabId = btn.getAttribute('data-tab');
            document.getElementById(`tab-${tabId}`).classList.add('active');

            if (tabId === 'graph') {
                const userId = document.getElementById('user_id').value || 'USER_RING1_1';
                renderGraphTopology(userId);
            } else if (tabId === 'logs') {
                refreshLogStream();
            } else if (tabId === 'transactions') {
                loadRecentTransactions();
            } else if (tabId === 'agent') {
                loadAgentStatus();
            }
        });
    });
}

// Preset Scenario Handlers
function loadPreset(type) {
    if (type === 'normal') {
        document.getElementById('user_id').value = 'USER_0042';
        document.getElementById('device_id').value = 'DEV_0088';
        document.getElementById('ip_address').value = '192.168.12.45';
        document.getElementById('amount').value = '1250';
        document.getElementById('velocity_1h').value = '1';
        document.getElementById('merchant_id').value = 'MCH_005';
        document.getElementById('is_vpn_proxy').checked = false;
        document.getElementById('is_suspicious_proxy').checked = false;
    } else if (type === 'ring1') {
        document.getElementById('user_id').value = 'USER_RING1_1';
        document.getElementById('device_id').value = 'DEV_FRAUD_RING1';
        document.getElementById('ip_address').value = '185.220.101.44';
        document.getElementById('amount').value = '88000';
        document.getElementById('velocity_1h').value = '12';
        document.getElementById('merchant_id').value = 'MCH_042';
        document.getElementById('is_vpn_proxy').checked = true;
        document.getElementById('is_suspicious_proxy').checked = true;
    } else if (type === 'ring2') {
        document.getElementById('user_id').value = 'USER_RING2_1';
        document.getElementById('device_id').value = 'DEV_RING2_USER_RING2_1';
        document.getElementById('ip_address').value = '198.51.100.99';
        document.getElementById('amount').value = '95000';
        document.getElementById('velocity_1h').value = '8';
        document.getElementById('merchant_id').value = 'MCH_042';
        document.getElementById('is_vpn_proxy').checked = true;
        document.getElementById('is_suspicious_proxy').checked = true;
    } else if (type === 'carding') {
        document.getElementById('user_id').value = 'USER_CARDER_X';
        document.getElementById('device_id').value = 'DEV_CARDER_X';
        document.getElementById('ip_address').value = '203.0.113.50';
        document.getElementById('amount').value = '49';
        document.getElementById('velocity_1h').value = '15';
        document.getElementById('merchant_id').value = 'MCH_012';
        document.getElementById('is_vpn_proxy').checked = true;
        document.getElementById('is_suspicious_proxy').checked = false;
    }
}

// Handle Transaction Scoring Submission
function handleTransactionScore(e) {
    e.preventDefault();
    const btn = document.getElementById('btn-score');
    btn.innerText = 'Scoring…';
    btn.disabled = true;

    const payload = {
        user_id: document.getElementById('user_id').value,
        device_id: document.getElementById('device_id').value,
        ip_address: document.getElementById('ip_address').value,
        amount: parseFloat(document.getElementById('amount').value),
        velocity_1h: parseInt(document.getElementById('velocity_1h').value),
        merchant_id: document.getElementById('merchant_id').value,
        is_vpn_proxy: document.getElementById('is_vpn_proxy').checked,
        is_suspicious_proxy: document.getElementById('is_suspicious_proxy').checked
    };

    fetch('/api/v1/transactions/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        btn.innerText = 'Score Transaction & Run AI Engine';
        btn.disabled = false;

        updateRiskDisplay(data.risk_evaluation);

        if (data.agent_investigation) {
            renderAgentReport(data.agent_investigation);
        } else {
            document.getElementById('agent-report-container').innerHTML = `
                <div class="placeholder-msg">
                    Risk score ${data.risk_evaluation.risk_score} is below the investigation threshold (70.0) — approved automatically, no agent report generated.
                </div>`;
        }

        loadStats();
    })
    .catch(err => {
        btn.innerText = 'Score Transaction & Run AI Engine';
        btn.disabled = false;
        alert('Could not evaluate this transaction: ' + err.message);
    });
}

function updateRiskDisplay(evalRes) {
    const score = evalRes.risk_score;
    const gauge = document.getElementById('risk-gauge');
    gauge.style.setProperty('--score-pct', `${score}%`);
    gauge.dataset.tier = (evalRes.risk_tier || 'LOW').toLowerCase();
    document.getElementById('score-val').innerText = score;

    const badge = document.getElementById('risk-tier-badge');
    badge.innerText = `${evalRes.risk_tier} RISK`;
    badge.className = `risk-badge tier-${(evalRes.risk_tier || 'low').toLowerCase()}`;

    document.getElementById('decision-text').innerText = `Decision: ${evalRes.decision}`;

    // Update Progress Bars
    document.getElementById('val-tabular').innerText = `${evalRes.tabular_score}%`;
    document.getElementById('bar-tabular').style.width = `${evalRes.tabular_score}%`;

    document.getElementById('val-gnn').innerText = `${evalRes.gnn_score}%`;
    document.getElementById('bar-gnn').style.width = `${evalRes.gnn_score}%`;

    document.getElementById('val-topology').innerText = `${evalRes.stacker_calibrated_score}%`;
    document.getElementById('bar-topology').style.width = `${evalRes.stacker_calibrated_score}%`;

    // Evidence
    document.getElementById('ev-devices').innerText = `${evalRes.graph_evidence.shared_device_accounts} accounts`;
    document.getElementById('ev-ips').innerText = `${evalRes.graph_evidence.shared_ip_accounts} accounts`;
    document.getElementById('ev-community').innerText = `${evalRes.graph_evidence.community_size} users`;
}

function renderAgentReport(agentRes) {
    const container = document.getElementById('agent-report-container');
    container.innerHTML = agentRes.summary_report;
    // Reflect the mode that ACTUALLY ran for this specific report — not
    // just the general "what would run next" status — so the badge never
    // implies more than the report it's sitting next to actually did.
    setAgentBadge(agentRes.agent_mode_label || agentRes.agent_mode, agentRes.agent_mode);
}

// Agent Mode Controls (Groq / Anthropic / OpenAI / deterministic-only)
function setAgentBadge(label, modeKey) {
    const badge = document.getElementById('agent-status-badge');
    if (!badge) return;
    badge.innerText = label;
    badge.className = 'agent-status-badge';
    if (modeKey && modeKey.indexOf('deterministic') !== -1) {
        badge.classList.add('status-deterministic');
    }
}

function loadAgentStatus() {
    fetch('/api/v1/investigations/agent-status')
        .then(res => res.json())
        .then(data => {
            const select = document.getElementById('agent-mode-select');
            if (select) {
                select.innerHTML = '';
                data.modes.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.value;
                    opt.innerText = m.available ? m.label : `${m.label} (no API key)`;
                    opt.disabled = !m.available;
                    if (m.value === data.current_mode) opt.selected = true;
                    select.appendChild(opt);
                });
            }
            setAgentBadge(data.active_label, data.active_provider || 'deterministic');
        })
        .catch(err => {
            console.error("Agent status fetch error:", err);
            setAgentBadge('Status unavailable', null);
            const badge = document.getElementById('agent-status-badge');
            if (badge) badge.classList.add('status-error');
        });
}

function setAgentMode(mode) {
    const select = document.getElementById('agent-mode-select');
    if (select) select.disabled = true;

    fetch('/api/v1/investigations/agent-mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode })
    })
        .then(async (res) => {
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Could not change agent mode');
            return data;
        })
        .then(data => {
            setAgentBadge(data.active_label, data.active_provider || 'deterministic');
        })
        .catch(err => {
            console.error("Agent mode change error:", err);
            alert("Couldn't change agent mode: " + err.message);
            loadAgentStatus(); // resync the dropdown with whatever the server actually has
        })
        .finally(() => {
            if (select) select.disabled = false;
        });
}

// Data Pipeline Controls (synthetic reseed / real Kaggle ingest + retrain)
function runDataPipeline(mode) {
    const statusEl = document.getElementById('pipeline-status');
    const synthBtn = document.getElementById('btn-seed-synthetic');
    const realBtn = document.getElementById('btn-seed-real');
    const endpoint = mode === 'real' ? '/api/v1/admin/pipeline/real' : '/api/v1/admin/pipeline/synthetic';
    const label = mode === 'real' ? 'Downloading real Kaggle dataset & retraining models…' : 'Regenerating synthetic data & retraining models…';

    statusEl.textContent = label;
    statusEl.className = 'pipeline-status status-loading';
    synthBtn.disabled = true;
    realBtn.disabled = true;

    fetch(endpoint, { method: 'POST' })
        .then(async (res) => {
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Pipeline failed');
            return data;
        })
        .then(data => {
            const count = data.transactions_generated ?? data.transactions_ingested ?? 0;
            const stacked = data.eval_metrics && data.eval_metrics.stacked;
            const metricsStr = stacked ? ` — held-out ROC-AUC ${stacked.roc_auc?.toFixed(3)}, precision ${stacked.precision?.toFixed(2)}, recall ${stacked.recall?.toFixed(2)}` : '';
            statusEl.textContent = `Done — ${count.toLocaleString()} transactions (${data.mode}), models retrained${metricsStr}.`;
            statusEl.className = 'pipeline-status status-ok';
            loadStats();
            loadRecentTransactions();
        })
        .catch(err => {
            statusEl.textContent = `Couldn't finish — ${err.message}`;
            statusEl.className = 'pipeline-status status-error';
            console.error('Data pipeline error:', err);
        })
        .finally(() => {
            synthBtn.disabled = false;
            realBtn.disabled = false;
            // Clear a settled status message after a while so it doesn't sit
            // in the header permanently — but leave errors up longer, since
            // those need to actually be read and acted on.
            const lingerMs = statusEl.className.includes('status-error') ? 12000 : 6000;
            setTimeout(() => {
                if (statusEl.textContent) statusEl.textContent = '';
                statusEl.className = 'pipeline-status';
            }, lingerMs);
        });
}

function loadStats() {
    fetch('/api/v1/stats')
        .then(res => res.json())
        .then(data => {
            document.getElementById('stat-total-txns').innerText = data.total_transactions;
            document.getElementById('stat-high-risk').innerText = data.high_risk_transactions;
            document.getElementById('stat-investigations').innerText = data.investigations_conducted;
        })
        .catch(err => console.error("Stats fetch error:", err));
}

function loadRecentTransactions() {
    fetch('/api/v1/transactions/recent?limit=15')
        .then(res => res.json())
        .then(data => {
            const tbody = document.getElementById('txns-table-body');
            tbody.innerHTML = '';
            if (data.transactions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8">No transactions logged yet.</td></tr>';
                return;
            }
            data.transactions.forEach(t => {
                const tr = document.createElement('tr');
                const tierClass = `tier-pill tier-${(t.risk_tier || 'unscored').toLowerCase()}`;
                tr.innerHTML = `
                    <td><code>${t.transaction_id}</code></td>
                    <td><code>${t.user_id}</code></td>
                    <td>${t.device_id}</td>
                    <td>${t.ip_address}</td>
                    <td>₹${t.amount.toLocaleString()}</td>
                    <td><strong>${t.risk_score}</strong></td>
                    <td><span class="${tierClass}">${t.risk_tier}</span></td>
                    <td><code>${t.decision}</code></td>
                `;
                tbody.appendChild(tr);
            });
        })
        .catch(err => console.error("Error loading txns:", err));
}

function refreshLogStream() {
    fetch('/api/v1/logs/stream?lines=25')
        .then(res => res.json())
        .then(data => {
            document.getElementById('log-risk-engine').innerText = data.risk_engine_logs.join('\n');
            document.getElementById('log-agent').innerText = data.agent_logs.join('\n');
        })
        .catch(err => console.error("Error streaming logs:", err));
}
