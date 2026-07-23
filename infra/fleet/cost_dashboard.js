const API_TOKEN = document.querySelector(
  'meta[name="api-token"]'
).content;
const C = {
  wait: '#c7ccd1',
  boot: '#4c78a8',
  steps: '#54a24b',
  shutdown: '#e45756'
};
const PALETTE = [
  '#4c78a8', '#f58518', '#54a24b', '#e45756', '#72b7b2',
  '#eeca3b', '#b279a2', '#ff9da6', '#9d755d', '#bab0ac',
  '#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c', '#98df8a',
  '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
];
const charts = {};
let accountRequestVersion = 0;

function chart(id) {
  if (!charts[id]) {
    charts[id] = echarts.init(document.getElementById(id));
  }
  return charts[id];
}

window.addEventListener('resize', () => {
  Object.values(charts).forEach(item => item.resize());
});

function escapeHTML(value) {
  return String(value ?? '').replace(
    /[&<>"']/g,
    character => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    })[character]
  );
}

function money(value) {
  return value == null ? 'unknown' : `$${value.toFixed(4)}`;
}

function mins(seconds) {
  return seconds == null ? null : seconds / 60;
}

function axisTooltip(parameters, suffix, digits = 1) {
  if (!parameters.length) {
    return '';
  }
  const lines = parameters
    .filter(item => item.value != null && item.seriesName !== 'off')
    .map(item => {
      const value = Number(item.value).toFixed(digits);
      return `${escapeHTML(item.marker)}${escapeHTML(item.seriesName)}: ` +
        `${value}${suffix}`;
    });
  return `<b>${escapeHTML(parameters[0].axisValueLabel)}</b><br>` +
    lines.join('<br>');
}

async function requestJSON(url, method = 'GET') {
  const response = await fetch(url, {
    method,
    cache: 'no-store',
    credentials: 'same-origin',
    headers: {[TOKEN_HEADER]: API_TOKEN}
  });
  const payload = await response.json();
  if (!response.ok || (payload && payload.error)) {
    throw new Error(payload.detail || payload.error || response.statusText);
  }
  return payload;
}

const TOKEN_HEADER = 'X-Cubie-Dashboard-Token';

function renderGantt(payload) {
  const legs = payload.legs;
  const categories = legs.map(leg => leg.label);
  const offsets = legs.map(leg => mins(leg.offset_s));
  const makeSeries = (name, key, color) => ({
    name,
    type: 'bar',
    stack: 'timeline',
    color,
    emphasis: {focus: 'series'},
    data: legs.map(leg => mins(leg[key]))
  });
  chart('cGantt').setOption({
    grid: {left: 150, right: 16, top: 30, bottom: 36},
    legend: {top: 0, data: ['wait', 'boot', 'CI steps', 'shutdown']},
    tooltip: {
      trigger: 'axis',
      axisPointer: {type: 'shadow'},
      formatter: parameters => axisTooltip(parameters, ' min')
    },
    xAxis: {
      type: 'value',
      name: 'minutes from first job scheduled',
      nameLocation: 'middle',
      nameGap: 24
    },
    yAxis: {
      type: 'category',
      data: categories,
      inverse: true,
      axisLabel: {fontSize: 10}
    },
    series: [
      {
        name: 'off',
        type: 'bar',
        stack: 'timeline',
        data: offsets,
        silent: true,
        itemStyle: {color: 'transparent'},
        tooltip: {show: false}
      },
      makeSeries('wait', 'wait_s', C.wait),
      makeSeries('boot', 'boot_s', C.boot),
      makeSeries('CI steps', 'steps_s', C.steps),
      makeSeries('shutdown', 'shutdown_s', C.shutdown)
    ]
  }, true);
}

function completeSum(legs, key) {
  if (legs.some(leg => leg[key] == null)) {
    return null;
  }
  return legs.reduce((total, leg) => total + leg[key], 0);
}

function renderGanttAggregate(payload) {
  const bands = [
    ['wait', 'wait_s', C.wait],
    ['boot', 'boot_s', C.boot],
    ['CI steps', 'steps_s', C.steps],
    ['shutdown', 'shutdown_s', C.shutdown]
  ];
  chart('cGanttAgg').setOption({
    grid: {left: 44, right: 10, top: 30, bottom: 36},
    legend: {show: false},
    tooltip: {
      trigger: 'item',
      formatter: item => item.value == null
        ? `${escapeHTML(item.seriesName)}: unknown`
        : `${escapeHTML(item.seriesName)}: ` +
          `${Number(item.value).toFixed(1)} min`
    },
    xAxis: {type: 'category', data: ['run total']},
    yAxis: {type: 'value', name: 'minutes'},
    series: bands.map(([name, key, color]) => ({
      name,
      type: 'bar',
      stack: 'timeline',
      color,
      data: [mins(completeSum(payload.legs, key))],
      label: {
        show: true,
        formatter: item => item.value != null && item.value >= 6 ? name : '',
        color: '#fff',
        fontSize: 10
      }
    }))
  }, true);
}

function stepTotals(legs) {
  const totals = {};
  legs.forEach(leg => leg.steps.forEach(step => {
    totals[step.name] = (totals[step.name] || 0) + step.dur_s;
  }));
  return Object.keys(totals).sort(
    (first, second) => totals[second] - totals[first]
  );
}

function renderSteps(payload) {
  const legs = payload.legs;
  const order = stepTotals(legs);
  const series = order.map((name, index) => ({
    name,
    type: 'bar',
    stack: 'steps',
    color: PALETTE[index % PALETTE.length],
    emphasis: {focus: 'series'},
    data: legs.map(leg => {
      const step = leg.steps.find(item => item.name === name);
      return step ? mins(step.dur_s) : 0;
    })
  }));
  chart('cSteps').setOption({
    grid: {left: 44, right: 16, top: 30, bottom: 96},
    legend: {type: 'scroll', top: 0, data: order},
    tooltip: {
      trigger: 'axis',
      axisPointer: {type: 'shadow'},
      formatter: parameters => axisTooltip(parameters, ' min', 2)
    },
    xAxis: {
      type: 'category',
      data: legs.map(leg => leg.label),
      axisLabel: {rotate: 40, fontSize: 10, interval: 0}
    },
    yAxis: {type: 'value', name: 'minutes'},
    series
  }, true);
}

function renderStepsAggregate(payload) {
  const order = stepTotals(payload.legs);
  const totals = {};
  payload.legs.forEach(leg => leg.steps.forEach(step => {
    totals[step.name] = (totals[step.name] || 0) + step.dur_s;
  }));
  chart('cStepsAgg').setOption({
    grid: {left: 44, right: 10, top: 30, bottom: 96},
    legend: {show: false},
    tooltip: {
      trigger: 'item',
      formatter: item => `${escapeHTML(item.seriesName)}: ` +
        `${Number(item.value).toFixed(2)} min`
    },
    xAxis: {type: 'category', data: ['run total']},
    yAxis: {type: 'value', name: 'minutes'},
    series: order.map((name, index) => ({
      name,
      type: 'bar',
      stack: 'steps',
      color: PALETTE[index % PALETTE.length],
      data: [mins(totals[name] || 0)]
    }))
  }, true);
}

const typeColors = {};

function colorFor(type) {
  if (!(type in typeColors)) {
    const index = Object.keys(typeColors).length % PALETTE.length;
    typeColors[type] = PALETTE[index];
  }
  return typeColors[type];
}

function renderCost(payload) {
  const legs = payload.legs;
  const types = [...new Set(
    legs.map(leg => leg.type).filter(Boolean)
  )].sort();
  const legend = document.getElementById('costLegend');
  legend.replaceChildren();
  types.forEach((type, index) => {
    if (index) {
      legend.append(document.createTextNode('   '));
    }
    const swatch = document.createElement('span');
    swatch.textContent = '■';
    swatch.style.color = colorFor(type);
    swatch.style.fontSize = '15px';
    legend.append(swatch, document.createTextNode(` ${type}`));
  });
  chart('cCost').setOption({
    grid: {left: 60, right: 16, top: 16, bottom: 96},
    tooltip: {
      trigger: 'item',
      formatter: item => {
        const leg = legs[item.dataIndex];
        return `${escapeHTML(leg.label)}<br>` +
          `${escapeHTML(leg.type || 'unknown type')} @ ` +
          `${money(leg.price)}/h<br><b>${money(leg.cost)}</b>`;
      }
    },
    xAxis: {
      type: 'category',
      data: legs.map(leg => leg.label),
      axisLabel: {rotate: 40, fontSize: 10, interval: 0}
    },
    yAxis: {type: 'value', name: 'USD (billed hrs × spot $/h)'},
    series: [{
      type: 'bar',
      data: legs.map(leg => ({
        value: leg.cost,
        itemStyle: {color: colorFor(leg.type || 'unknown')}
      })),
      label: {
        show: true,
        position: 'top',
        fontSize: 9,
        formatter: item => item.value == null
          ? ''
          : `$${Number(item.value).toFixed(3)}`
      }
    }]
  }, true);
}

function renderType(payload) {
  const aggregates = {};
  payload.legs.forEach(leg => {
    if (!leg.type) {
      return;
    }
    const aggregate = aggregates[leg.type] || {
      minutes: 0,
      cost: 0,
      hours: 0,
      complete: true
    };
    if (leg.billed_hours == null || leg.cost == null) {
      aggregate.complete = false;
    } else {
      aggregate.minutes += leg.billed_hours * 60;
      aggregate.cost += leg.cost;
      aggregate.hours += leg.billed_hours;
    }
    aggregates[leg.type] = aggregate;
  });
  const types = Object.keys(aggregates).sort();
  chart('cType').setOption({
    grid: {left: 56, right: 56, top: 30, bottom: 40},
    legend: {top: 0, data: ['minutes', 'cost']},
    tooltip: {
      trigger: 'axis',
      axisPointer: {type: 'shadow'},
      formatter: parameters => axisTooltip(parameters, '', 4)
    },
    xAxis: {type: 'category', data: types},
    yAxis: [
      {type: 'value', name: 'billed minutes'},
      {type: 'value', name: 'USD', position: 'right'}
    ],
    series: [
      {
        name: 'minutes',
        type: 'bar',
        color: C.boot,
        data: types.map(type => {
          const aggregate = aggregates[type];
          return aggregate.complete
            ? Number(aggregate.minutes.toFixed(1))
            : null;
        })
      },
      {
        name: 'cost',
        type: 'bar',
        color: C.steps,
        yAxisIndex: 1,
        data: types.map(type => {
          const aggregate = aggregates[type];
          return aggregate.complete
            ? Number(aggregate.cost.toFixed(4))
            : null;
        }),
        label: {
          show: true,
          position: 'top',
          fontSize: 10,
          formatter: item => {
            const aggregate = aggregates[types[item.dataIndex]];
            return aggregate.complete && aggregate.hours
              ? `~$${(aggregate.cost / aggregate.hours).toFixed(3)}/h`
              : '';
          }
        }
      }
    ]
  }, true);
}

function renderWait(payload) {
  const legs = payload.legs;
  chart('cWait').setOption({
    grid: {left: 56, right: 16, top: 16, bottom: 96},
    tooltip: {
      trigger: 'axis',
      axisPointer: {type: 'shadow'},
      formatter: parameters => axisTooltip(parameters, ' min')
    },
    xAxis: {
      type: 'category',
      data: legs.map(leg => leg.label),
      axisLabel: {rotate: 40, fontSize: 10, interval: 0}
    },
    yAxis: {type: 'value', name: 'minutes'},
    series: [{
      type: 'bar',
      color: C.wait,
      data: legs.map(leg => Number(mins(leg.wait_s).toFixed(2)))
    }]
  }, true);
}

function renderRun(payload) {
  const empty = document.getElementById('runEmpty');
  const chartArea = document.getElementById('runCharts');
  if (!payload.legs.length) {
    document.getElementById('runMeta').textContent =
      `run ${payload.run_id} · 0 GPU legs`;
    empty.textContent =
      'This run did not launch any completed GPU legs. Select another run.';
    empty.classList.remove('hidden');
    chartArea.classList.add('hidden');
    return;
  }
  empty.classList.add('hidden');
  chartArea.classList.remove('hidden');
  const known = payload.legs.filter(leg => leg.cost != null);
  const knownCost = known.reduce((total, leg) => total + leg.cost, 0);
  const missing = payload.legs.length - known.length;
  const costText = missing
    ? `known compute cost $${knownCost.toFixed(3)}; ` +
      `${missing} leg${missing === 1 ? '' : 's'} missing spot price ` +
      'or termination telemetry'
    : `compute cost $${knownCost.toFixed(3)}`;
  document.getElementById('runMeta').textContent =
    `run ${payload.run_id} · ${payload.legs.length} GPU legs · ${costText}`;
  renderGantt(payload);
  renderGanttAggregate(payload);
  renderSteps(payload);
  renderStepsAggregate(payload);
  renderCost(payload);
  renderType(payload);
  renderWait(payload);
}

function renderStack(id, times, series, yName) {
  const wrapper = document.getElementById(id).parentElement;
  if (!series.length) {
    const note = document.createElement('div');
    note.className = 'note';
    note.textContent =
      'No available Cost Explorer data has a non-zero value in this window.';
    wrapper.replaceChildren(note);
    return;
  }
  chart(id).setOption({
    grid: {left: 56, right: 16, top: 30, bottom: 70},
    legend: {type: 'scroll', top: 0},
    tooltip: {
      trigger: 'axis',
      axisPointer: {type: 'shadow'},
      formatter: parameters => axisTooltip(parameters, '', 4)
    },
    xAxis: {
      type: 'category',
      data: times.map(time => time
        .replace('T', ' ')
        .replace(':00:00Z', 'h')
        .slice(5)),
      axisLabel: {rotate: 60, fontSize: 9, interval: 0}
    },
    yAxis: {type: 'value', name: yName},
    series: series.map((item, index) => ({
      name: item.name,
      type: 'bar',
      stack: 'account',
      color: PALETTE[index % PALETTE.length],
      data: item.data
    }))
  }, true);
}

function resetAccountChart(wrapperId, chartId) {
  if (charts[chartId]) {
    charts[chartId].dispose();
    delete charts[chartId];
  }
  const chartElement = document.createElement('div');
  chartElement.id = chartId;
  chartElement.className = 'chart';
  document.getElementById(wrapperId).replaceChildren(chartElement);
}

async function loadAccount(force) {
  const requestVersion = ++accountRequestVersion;
  const status = document.getElementById('acctStatus');
  const start = document.getElementById('acctFrom').value;
  const end = document.getElementById('acctTo').value;
  const granularity = document.getElementById('acctGran').value;
  const query = new URLSearchParams({
    start,
    end,
    gran: granularity
  });
  status.textContent = 'fetching…';
  try {
    const route = force ? '/api/account/refresh' : '/api/account';
    const payload = await requestJSON(
      `${route}?${query.toString()}`,
      force ? 'POST' : 'GET'
    );
    if (requestVersion !== accountRequestVersion) return;
    resetAccountChart('acctUsageWrap', 'cAcctUsage');
    resetAccountChart('acctCostWrap', 'cAcctCost');
    renderStack('cAcctUsage', payload.times, payload.usage, 'EC2 hours');
    renderStack('cAcctCost', payload.times, payload.cost, 'USD (gross usage)');
    const frontier = payload.frontier
      ? payload.frontier
        .replace('T', ' ')
        .replace(':00:00Z', 'h UTC')
      : '—';
    const fetched = payload.last_fetch
      ? new Date(payload.last_fetch).toLocaleString()
      : '—';
    const statusLabels = {
      fetched: 'fetched from Cost Explorer',
      cached: 'served from local SQLite',
      in_progress: 'another refresh is in progress; served stored data',
      rate_limited: 'force fetch recently attempted; served stored data'
    };
    const fetchLabel = payload.charged
      ? 'fetched from Cost Explorer'
      : statusLabels[payload.refresh_status] || payload.refresh_status;
    status.textContent =
      `${fetchLabel}` +
      ` · last non-zero usage ${frontier} · last fetch ${fetched}`;
    const warning = document.getElementById('acctWarning');
    if (payload.coverage.warning) {
      warning.textContent = payload.coverage.warning;
      warning.classList.remove('hidden');
    } else {
      warning.textContent = '';
      warning.classList.add('hidden');
    }
  } catch (error) {
    if (requestVersion === accountRequestVersion) {
      status.textContent = `error: ${error.message}`;
    }
  }
}

async function loadRun(id) {
  const status = document.getElementById('status');
  status.textContent = `fetching run ${id} …`;
  try {
    const payload = await requestJSON(`/api/run?id=${encodeURIComponent(id)}`);
    renderRun(payload);
    status.textContent = '';
    return payload;
  } catch (error) {
    status.textContent = `error: ${error.message}`;
    return null;
  }
}

async function loadFirstLegBearingRun(runs) {
  const selector = document.getElementById('runSelect');
  for (const run of runs) {
    const payload = await loadRun(run.id);
    if (payload && payload.legs.length) {
      selector.value = String(run.id);
      return;
    }
  }
  if (runs.length) {
    selector.value = String(runs[0].id);
    await loadRun(runs[0].id);
  } else {
    document.getElementById('runMeta').textContent = 'No recent runs found.';
    document.getElementById('runCharts').classList.add('hidden');
  }
}

async function init() {
  const iso = value => value.toISOString().slice(0, 10);
  const now = new Date();
  document.getElementById('acctTo').value = iso(now);
  document.getElementById('acctFrom').value = iso(
    new Date(now.getTime() - 29 * 864e5)
  );
  document.getElementById('acctFrom').onchange = () => loadAccount(false);
  document.getElementById('acctTo').onchange = () => loadAccount(false);
  document.getElementById('acctGran').onchange = () => loadAccount(false);
  document.getElementById('acctForce').onclick = () => loadAccount(true);
  loadAccount(false);
  const selector = document.getElementById('runSelect');
  try {
    const runs = await requestJSON('/api/runs');
    selector.replaceChildren();
    runs.forEach(run => {
      const option = document.createElement('option');
      option.value = run.id;
      option.textContent =
        `${run.created_at.slice(0, 16).replace('T', ' ')} · ` +
        `${run.event} · ${run.conclusion || run.status}`;
      selector.appendChild(option);
    });
    selector.onchange = () => loadRun(selector.value);
    const parameters = new URLSearchParams(location.search);
    const selected = parameters.get('run');
    if (selected) {
      selector.value = selected;
      await loadRun(selected);
    } else {
      await loadFirstLegBearingRun(runs);
    }
  } catch (error) {
    selector.replaceChildren(new Option('error'));
    document.getElementById('status').textContent = error.message;
  }
}

init();
