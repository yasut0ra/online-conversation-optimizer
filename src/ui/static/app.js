
(function () {
  const state = {
    history: [],
    sessionId: null,
    pendingUser: null,
    lastTurnResponseAt: null,
  };

  const conversation = document.getElementById('conversation');
  const candidatesBox = document.getElementById('candidates');
  const historyInput = document.getElementById('history-json');
  const sessionInput = document.getElementById('session-id');
  const messageInput = document.getElementById('user-utterance');
  const form = document.getElementById('turn-form');
  const statusMessage = document.getElementById('status-message');

  function ensureSessionId() {
    if (sessionInput.value) {
      state.sessionId = sessionInput.value;
      return;
    }
    const fallback = 'sess-' + Date.now();
    const id = (self.crypto && self.crypto.randomUUID) ? self.crypto.randomUUID() : fallback;
    state.sessionId = id;
    sessionInput.value = id;
  }

  function updateHistoryInput() {
    historyInput.value = JSON.stringify(state.history);
  }

  function appendMessage(role, text) {
    const wrapper = document.createElement('div');
    wrapper.className = 'message ' + role + ' rounded-md border border-slate-800 bg-slate-900/60 px-4 py-3';
    const label = document.createElement('div');
    label.className = 'text-xs text-slate-400 mb-1';
    label.textContent = role === 'user' ? 'ユーザ' : 'アシスタント';
    const body = document.createElement('p');
    body.className = 'whitespace-pre-line text-slate-100';
    body.textContent = text;
    wrapper.appendChild(label);
    wrapper.appendChild(body);
    conversation.appendChild(wrapper);
    conversation.scrollTop = conversation.scrollHeight;
    return wrapper;
  }

  function removeLastUserMessage() {
    const last = conversation.lastElementChild;
    if (!last) {
      return;
    }
    if (last.className && last.className.indexOf('message user') !== -1) {
      conversation.removeChild(last);
    }
  }

  function handleCandidateClick(event) {
    const card = event.target.closest('.candidate-card');
    if (!card || card.dataset.disabled === 'true') {
      return;
    }
    const candidate = JSON.parse(card.dataset.candidate);
    const latencyMs = state.lastTurnResponseAt ? Math.round(performance.now() - state.lastTurnResponseAt) : null;
    const payload = {
      session_id: candidate.session_id,
      turn_id: candidate.turn_id,
      chosen_idx: candidate.index,
      reward: 1.0,
      latency_ms: latencyMs,
      continued: true,
    };
    card.dataset.disabled = 'true';
    fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
      .then(function (response) {
        if (!response.ok) {
          throw new Error('feedback failed');
        }
        return response.json();
      })
      .then(function () {
        appendMessage('assistant', candidate.text);
        if (state.pendingUser !== null) {
          state.history.push(state.pendingUser);
        }
        state.history.push(candidate.text);
        state.pendingUser = null;
        messageInput.value = '';
        candidatesBox.innerHTML = '';
        updateHistoryInput();
        statusMessage.textContent = '';
        refreshMetrics();
      })
      .catch(function () {
        statusMessage.textContent = 'フィードバックに失敗しました';
      })
      .finally(function () {
        delete card.dataset.disabled;
      });
  }

  document.addEventListener('DOMContentLoaded', function () {
    ensureSessionId();
    updateHistoryInput();
    refreshMetrics();
    setInterval(refreshMetrics, 5000);
    candidatesBox.addEventListener('click', handleCandidateClick);
  });

  form.addEventListener('submit', function (event) {
    const message = messageInput.value.trim();
    if (!message) {
      event.preventDefault();
      statusMessage.textContent = '入力してください';
      return;
    }
    state.pendingUser = message;
    updateHistoryInput();
    appendMessage('user', message);
    candidatesBox.innerHTML = '';
    statusMessage.textContent = '候補を生成中...';
  });

  document.body.addEventListener('htmx:afterSwap', function (event) {
    if (event.target === candidatesBox) {
      statusMessage.textContent = '';
      state.lastTurnResponseAt = performance.now();
      const list = candidatesBox.querySelector('#candidate-list');
      if (!list) {
        return;
      }
      const sessionId = list.dataset.sessionId;
      const turnId = list.dataset.turnId;
      list.querySelectorAll('.candidate-card').forEach(function (card) {
        const candidate = JSON.parse(card.dataset.candidate);
        candidate.session_id = sessionId;
        candidate.turn_id = turnId;
        card.dataset.candidate = JSON.stringify(candidate);
      });
    }
  });

  document.body.addEventListener('htmx:responseError', function (event) {
    if (event.target === form) {
      statusMessage.textContent = '候補生成に失敗しました';
      removeLastUserMessage();
      state.pendingUser = null;
    }
  });

  function renderMetrics(data) {
    document.getElementById('metric-turn-count').textContent = data.turn_count != null ? data.turn_count : '-';
    const avg = data.avg_reward;
    document.getElementById('metric-avg-reward').textContent =
      avg !== null && avg !== undefined ? avg.toFixed(3) : 'n/a';
    const exploration = data.exploration_rate != null ? data.exploration_rate : 0;
    document.getElementById('metric-exploration').textContent = Math.round(exploration * 100) + '%';
    const meanValue = data.propensity_mean;
    const stdValue = data.propensity_std;
    if (meanValue !== null && meanValue !== undefined) {
      const stdPart = (stdValue !== null && stdValue !== undefined) ? ' / σ=' + stdValue.toFixed(3) : '';
      document.getElementById('metric-propensity').textContent = meanValue.toFixed(3) + stdPart;
    } else {
      document.getElementById('metric-propensity').textContent = 'n/a';
    }
    const list = document.getElementById('metric-style-rates');
    list.innerHTML = '';
    if (data.style_win_rates) {
      Object.entries(data.style_win_rates)
        .sort(function (a, b) { return b[1] - a[1]; })
        .forEach(function (entry) {
          const style = entry[0];
          const rate = entry[1];
          const item = document.createElement('li');
          item.textContent = style + ': ' + (rate * 100).toFixed(1) + '%';
          list.appendChild(item);
        });
    }
  }

  function refreshMetrics() {
    fetch('/metrics')
      .then(function (response) { return response.ok ? response.json() : null; })
      .then(function (data) {
        if (!data) {
          return;
        }
        renderMetrics(data);
      })
      .catch(function () {});
  }
})();
