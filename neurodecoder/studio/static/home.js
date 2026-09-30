// Homepage: choose a session. The page sends filters and draws what the server returns;
// filtering, counts, sorting and the region tree are all computed server-side.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const state = { region: '', sort: 'date', desc: false, options: null, seq: 0 };
const COLUMNS = [
  ['lab', 'Lab'], ['subject', 'Subject'], ['date', 'Date'], ['n_probes', 'Probes', 'num'],
  ['n_good_units', 'Good units*', 'num'], ['n_trials', 'Trials', 'num'],
  ['n_included_trials', 'Included', 'num'], ['n_regions', 'Beryl regions', 'num'],
];

// ---------- theme (same behaviour as the session view) ----------
function setTheme(v) {
  if (v === 'auto') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = v;
  for (const b of $('theme').querySelectorAll('button')) b.setAttribute('aria-pressed', b.dataset.v === v);
  try { localStorage.setItem('studio-theme', v); } catch { /* storage may be unavailable */ }
}
$('theme').addEventListener('click', (e) => { if (e.target.dataset.v) setTheme(e.target.dataset.v); });
try { const t = localStorage.getItem('studio-theme'); if (t) setTheme(t); } catch { /* ignore */ }

// ---------- filters -> request ----------
function sessionFilter() {
  const f = {};
  if ($('lab').value) f.labs = [$('lab').value];
  if ($('subject').value) f.subjects = [$('subject').value];
  if ($('dateFrom').value) f.date_from = $('dateFrom').value;
  if ($('dateTo').value) f.date_to = $('dateTo').value;
  if (+$('minUnits').value) f.min_good_units = +$('minUnits').value;
  if (+$('minTrials').value) f.min_included_trials = +$('minTrials').value;
  if ($('nProbes').value) f.n_probes = [+$('nProbes').value];
  const mods = [...$('modalities').querySelectorAll('input:checked')].map((i) => i.value);
  if (mods.length) f.modalities = mods;
  if (state.region) {
    f.region = state.region;
    f.min_region_units = Math.max(1, +$('minRegionUnits').value || 1);
  }
  return f;
}
function checked(id) {
  const boxes = [...$(id).querySelectorAll('input')];
  const on = boxes.filter((i) => i.checked).map((i) => +i.value);
  return on.length === boxes.length ? [] : on;  // all ticked means no filter
}
function trialFilter() {
  return {
    bwm_include: $('tfInclude').checked, exclude_nogo: $('tfNogo').checked,
    contrasts: checked('tfContrasts'), blocks: checked('tfBlocks'), outcomes: checked('tfOutcomes'),
  };
}

// ---------- load and draw ----------
async function refresh() {
  const seq = ++state.seq;
  const p = new URLSearchParams({ f: JSON.stringify(sessionFilter()), sort: state.sort, desc: state.desc ? 1 : 0 });
  const r = await fetch('/api/home?' + p);
  if (seq !== state.seq) return;  // a newer request replaced this one
  if (!r.ok) { $('counts').textContent = await r.text(); return; }
  const d = await r.json();
  if (!state.options) setup(d);
  const s = d.summary;
  $('counts').textContent = `${s.n_sessions} sessions, ${s.n_probes} probes, ${s.n_units.toLocaleString()} units match` +
    (s.n_region_units == null ? '' : ` · ${s.n_region_units.toLocaleString()} of those units in ${state.region}`);
  $('unitsNote').textContent = `* ${d.notes.units} Manifest v${d.notes.manifest.manifest_version}.`;
  $('backToSession').hidden = !d.open;
  drawTree(d.tree);
  drawTable(d.sessions);
}

function setup(d) {
  const o = (state.options = d.options);
  for (const v of o.labs) $('lab').add(new Option(v, v));
  for (const v of o.subjects) $('subject').add(new Option(v, v));
  for (const v of o.n_probes) $('nProbes').add(new Option(String(v), v));
  $('dateFrom').min = $('dateTo').min = o.dates[0];
  $('dateFrom').max = $('dateTo').max = o.dates[1];
  $('minRegionUnits').value = o.min_region_units;
  $('modalities').innerHTML = o.modalities.map((m) => `<label><input type="checkbox" value="${esc(m)}"> ${esc(m)}</label>`).join('');
  const t = o.default_trial_filter, lv = o.trial_levels;
  $('tfInclude').checked = t.bwm_include;
  $('tfNogo').checked = t.exclude_nogo;
  const boxes = (id, values, chosen, name) => {
    $(id).innerHTML = values.map((v) => `<label><input type="checkbox" value="${v}" ${!chosen.length || chosen.includes(v) ? 'checked' : ''}> ${esc(name(v))}</label>`).join('');
  };
  boxes('tfContrasts', lv.contrasts, t.contrasts, (v) => `${v * 100}%`);
  boxes('tfBlocks', lv.blocks, t.blocks, (v) => `p(left) ${v}`);
  boxes('tfOutcomes', lv.outcomes, t.outcomes, (v) => (v < 0 ? 'error' : 'reward'));
  $('heads').innerHTML = COLUMNS.map(([k, label, cls]) => `<th class="sortable ${cls || ''}" data-k="${k}">${label}</th>`).join('') + '<th></th><th></th>';
}

function drawTree(tree) {
  const top = `<div class="node" data-node="" aria-selected="${!state.region}"><span>Any region</span></div>`;
  $('tree').innerHTML = top + tree.map((n) =>
    `<div class="node" data-node="${esc(n.acronym)}" aria-selected="${state.region === n.acronym}" style="padding-left:${6 + n.depth * 10}px" title="${esc(n.name)}">` +
    `<span class="sw" style="background:${n.colour}"></span><span>${esc(n.acronym)}</span><span class="n">${n.n_total.toLocaleString()}</span></div>`).join('');
  $('regionNote').textContent = state.region
    ? `Sessions with at least ${$('minRegionUnits').value} good units in ${state.region} or below it.`
    : 'No region chosen. Pick one to require units there, descendants included.';
}

function drawTable(rows) {
  for (const th of $('heads').querySelectorAll('th[data-k]')) {
    if (th.dataset.k === state.sort) th.setAttribute('aria-sort', state.desc ? 'descending' : 'ascending');
    else th.removeAttribute('aria-sort');
  }
  $('rows').innerHTML = rows.map((r) => `<tr data-eid="${esc(r.eid)}">` +
    COLUMNS.map(([k, , cls]) => `<td class="${cls || ''}" ${k === 'n_regions' ? `title="${esc(r.regions.join(', '))}"` : ''}>${esc(r[k])}</td>`).join('') +
    `<td title="${r.cached ? 'in the local cache' : 'loads from the release'}">${r.cached ? '<span class="dot">●</span>' : ''}</td>` +
    `<td class="open-cell"><button class="btn" data-open="${esc(r.eid)}">Open</button></td></tr>`).join('');
}

// ---------- opening a session ----------
async function openSession(eid) {
  $('loadingText').textContent = `Loading session ${eid.slice(0, 8)}… from the cache if it's there, else from the release.`;
  $('loading').hidden = false;
  try {
    const r = await fetch('/api/open', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind: 'ibl', eid, trials: trialFilter() }),
    });
    if (!r.ok) throw new Error(await r.text());
    window.location = (await r.json()).url;
  } catch (e) {
    $('loadingText').textContent = e.message;
    setTimeout(() => { $('loading').hidden = true; }, 4000);
  }
}

// ---------- wiring ----------
for (const id of ['lab', 'subject', 'dateFrom', 'dateTo', 'minUnits', 'minTrials', 'nProbes', 'minRegionUnits']) {
  $(id).addEventListener('change', refresh);
}
$('modalities').addEventListener('change', refresh);
$('tree').addEventListener('click', (e) => {
  const n = e.target.closest('.node');
  if (!n) return;
  state.region = n.dataset.node;
  refresh();
});
$('heads').addEventListener('click', (e) => {
  const th = e.target.closest('th[data-k]');
  if (!th) return;
  state.desc = state.sort === th.dataset.k ? !state.desc : false;
  state.sort = th.dataset.k;
  refresh();
});
$('rows').addEventListener('click', (e) => { const b = e.target.closest('[data-open]'); if (b) openSession(b.dataset.open); });
refresh();
