// Neurodecoder Studio page. It draws what the server sends and computes no numbers:
// regions, trees, positions, PSTHs and heatmap row order all come from the API.
import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/three/OrbitControls.js';
import { OBJLoader } from '/static/vendor/three/OBJLoader.js';

const $ = (id) => document.getElementById(id);
const state = { session: null, level: null, node: '', all: false, responsive: false, movementFree: false, probe: '', split: '', trials: {}, unit: null, rows: [], tree: [], popRows: null, box: null };

// ---------- helpers ----------
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
function fmt(v, digits) {
  if (v === null || v === undefined) return '<span class="missing">—</span>';  // missing, never a number
  return typeof v === 'number' ? v.toFixed(digits) : esc(v);
}
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
function params(extra = {}) {
  const p = new URLSearchParams({
    event: $('event').value, t0: $('t0').value, t1: $('t1').value, bin: $('bin').value,
    baseline: $('baseline').checked ? 1 : 0, b0: $('b0').value, b1: $('b1').value,
    all: state.all ? 1 : 0, node: state.node, probe: state.probe, split: state.split, tf: JSON.stringify(state.trials),
    responsive: state.responsive ? 1 : 0, movement_free: state.movementFree ? 1 : 0, theme: theme(), ...extra,
  });
  if (state.level) p.set('level', state.level);
  return p;
}
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
// Allen's root is white; give it the muted ink so it stays visible on any surface.
const visible = (colour) => (!colour || colour === '#ffffff' ? css('--muted') : colour);

// ---------- theme ----------
function theme() {
  const t = document.documentElement.dataset.theme;
  return t || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
}
function setTheme(v) {
  if (v === 'auto') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = v;
  for (const b of $('theme').querySelectorAll('button')) b.setAttribute('aria-pressed', b.dataset.v === v);
  try { localStorage.setItem('studio-theme', v); } catch { /* storage may be unavailable */ }
  redrawForTheme();
}
function redrawForTheme() {
  if (!state.session) return;  // nothing drawn yet
  renderProbes();
  plotUnit(); plotPop(); renderProbe(); brain.theme();
}
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  if (!document.documentElement.dataset.theme) redrawForTheme();
});

// ---------- data ----------
async function init() {
  const s = (state.session = await getJSON('/api/session'));
  const phy = s.eid.startsWith('phy:');
  $('source').textContent = phy ? `Phy folder · ${s.eid.slice(4)}` : `IBL session · ${s.eid}`;
  $('source').title = s.eid;
  $('counts').textContent = `${s.n_units_passing} of ${s.n_units_total} units pass QC · ${s.n_trials} trials`;
  for (const [k, v] of Object.entries(s.events)) $('event').add(new Option(v, k));
  for (const [k, v] of Object.entries(s.conditions)) $('split').add(new Option(v, k));
  const noRegion = s.missing['units.acronym'];
  // Start from the project's saved view: settings only; every number is recomputed.
  const v = s.project.view;
  if (s.events[v.event]) $('event').value = v.event;
  for (const k of ['t0', 't1', 'bin', 'b0', 'b1']) $(k).value = v[k];
  $('baseline').checked = v.baseline;
  $('all').checked = state.all = v.all;
  state.node = v.node || '';
  state.unit = v.unit;
  state.probe = s.probes.includes(v.probe) ? v.probe : '';
  state.split = s.conditions[v.split] ? v.split : '';
  state.trials = v.trials || {};
  state.movementFree = Boolean(v.movement_free) && s.movement.first_movement;
  $('movementFree').checked = state.movementFree;
  renderTrialFilters();
  $('split').value = state.split;
  renderProbes();
  state.level = noRegion ? null : (s.levels.includes(v.level) ? v.level : s.default_level);
  $('projectStatus').textContent = s.project.path ? `Project: ${s.project.path.split('/').pop()}` : '';
  $('projectStatus').title = s.project.path || '';
  renderWarnings(s.project.warnings);
  $('level').innerHTML = s.levels
    .map((l) => `<button data-v="${l}" aria-pressed="${l === state.level}" ${noRegion ? 'disabled' : ''}>${l}</button>`)
    .join('');
  $('levelNote').textContent = noRegion || '';
  const r = s.response, ms = (w) => `${w[0] * 1000} to ${w[1] * 1000} ms`;
  $('testWindows').textContent = `Response ${ms(r.response_window)} vs baseline ${ms(r.baseline_window)}, ` +
    `two-sided, against every circular shift of each spike train (≥ ${r.min_shift_s} s). ` +
    `Benjamini–Hochberg across the units tested, α = ${r.alpha}.`;
  renderMovementNotes();
  await loadUnits();
  if (v.responsive) {  // results are never saved: rerun the test the view relied on
    await runTest();
    state.responsive = $('responsive').checked = true;
    await loadUnits();
  }
  loadGeometry();
}

function renderWarnings(list) {
  const el = $('warnings');
  el.hidden = !list.length;
  el.innerHTML = list.length
    ? `<strong>⚠ Changed since this project was saved</strong><ul>${list.map((w) => `<li>${esc(w)}</li>`).join('')}</ul>`
    : '';
}

// ---------- project and export ----------
function currentView() {
  return {
    event: $('event').value, t0: +$('t0').value, t1: +$('t1').value, bin: +$('bin').value,
    baseline: $('baseline').checked, b0: +$('b0').value, b1: +$('b1').value,
    level: state.level || state.session.default_level, node: state.node,
    all: state.all, responsive: state.responsive, unit: state.unit, probe: state.probe, split: state.split,
    trials: state.trials, movement_free: state.movementFree,
  };
}
async function post(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
async function act(button, busy, work) {
  button.disabled = true;
  $('projectStatus').textContent = busy;
  try { $('projectStatus').textContent = await work(); }
  catch (e) { $('projectStatus').textContent = e.message; }
  finally { button.disabled = false; }
}

async function loadUnits() {
  const d = await getJSON('/api/units?' + params());
  state.rows = d.units;
  state.tree = d.tree;
  renderTest(d.test);
  renderSelectivity(d.selectivity);
  renderLocking(d.locking);
  renderTrialCounts(d.trials);
  if (!state.rows.some((u) => u.id === state.unit)) state.unit = state.rows.length ? state.rows[0].id : null;
  renderTree();
  renderTable();
  selectUnit(state.unit, { scroll: false });
  plotPop();
}

async function loadGeometry() {
  try {
    await brain.load(await getJSON('/api/geometry?' + params()));
    brain.select(state.unit);
  } catch (e) {
    brain.note(e.message);
  }
}

// ---------- rail: probe filter ----------
// Colours come from the server, fixed by each probe's place in the session.
function renderProbes() {
  const s = state.session, colours = s.probe_colours[theme()];
  const button = (value, label, colour) =>
    `<button data-v="${esc(value)}" aria-pressed="${state.probe === value}">` +
    (colour ? `<span class="sw" style="background:${colour}"></span>` : '') + `${esc(label)}</button>`;
  $('probeFilter').innerHTML = button('', 'All probes') + s.probes.map((p) => button(p, p, colours[p])).join('');
}

// ---------- rail: responsiveness ----------
function renderTest(t) {
  $('responsive').disabled = !t;
  if (!t) {
    $('testSummary').textContent = `Not run for ${$('event').selectedOptions[0]?.text || 'this event'}.`;
    return;
  }
  const on = t.probes.length === 1 ? `probe ${t.probes[0]}` : `probes ${t.probes.join(', ')}`;
  $('testSummary').textContent = `${t.n_responsive} of ${t.n_tests} units responsive on ${on} (${t.n_up} up, ${t.n_down} down) · ` +
    `${t.n_trials} ${t.movement_free ? 'movement-free ' : ''}trials${t.n_excluded ? `, ${t.n_excluded} without this event excluded` : ''} · ${t.n_shifts.toLocaleString()} shifts each`;
}
async function runTest() {
  const button = $('runTest');
  button.disabled = true;
  $('testSummary').textContent = 'Testing… about 10 s for 400 units.';
  try {
    await getJSON('/api/test?' + params({ responsive: 0 }));
    await loadUnits();
  } catch (e) {
    $('testSummary').textContent = e.message;
  } finally {
    button.disabled = false;
  }
}
// A test belongs to one event and one unit set; changing either drops the filter.
function resetResponsive() { state.responsive = false; $('responsive').checked = false; }

// ---------- rail: movement ----------
// Movement-free trials and the locking test need each trial's first movement; the
// notes say what each does, or why it is unavailable. The server does the work.
function renderMovementNotes() {
  const m = state.session.movement, r = state.session.response, stim = $('event').value === 'stim_on';
  const ms = (w) => `${w[0] * 1000} to ${w[1] * 1000} ms`;
  $('movementFree').disabled = !m.first_movement || !stim;
  $('freeBox').classList.toggle('off', $('movementFree').disabled);
  $('freeNote').textContent = !m.first_movement ? 'No first-movement times in this session.'
    : !stim ? 'Defined at stimulus onset only.'
      : `Tests only trials whose first movement comes after ${r.response_window[1] * 1000} ms, the end of the response window. Plots keep every trial.`;
  $('runLock').disabled = !m.first_movement;
  $('lockWhat').textContent = !m.first_movement ? 'No first-movement times in this session.'
    : `Rate ${ms(m.windows[1])} minus ${ms(m.windows[0])} around each trial's own first movement, ` +
      `against reaction times permuted within signed contrast. Benjamini–Hochberg across the units tested, α = ${r.alpha}.`;
}
function renderLocking(t) {
  if (!state.session.movement.first_movement) { $('lockSummary').textContent = ''; return; }
  if (!t) { $('lockSummary').textContent = 'Not run for these units and trials.'; return; }
  const on = t.probes.length === 1 ? `probe ${t.probes[0]}` : `probes ${t.probes.join(', ')}`;
  $('lockSummary').textContent = `${t.n_locked} of ${t.n_tests} units on ${on} movement-locked · ${t.n_trials} trials · ` +
    `null: ${t.null}, ${t.n_null.toLocaleString()} draws, seed ${t.seed}`;
}
async function runLocking() {
  const button = $('runLock');
  button.disabled = true;
  $('lockSummary').textContent = 'Testing… about 30 s for 400 units.';
  try {
    await getJSON('/api/locking?' + params({ responsive: 0 }));
    await loadUnits();
  } catch (e) {
    $('lockSummary').textContent = e.message;
  } finally {
    button.disabled = !state.session.movement.first_movement;
  }
}

// ---------- rail: trial filters ----------
// Set on the homepage (or in the project); changeable here. The server applies them.
function renderTrialFilters() {
  const s = state.session, t = state.trials, offered = s.trial_filters, lv = s.trial_levels;
  const off = (name) => (offered[name].available ? '' : `disabled title="${esc(offered[name].reason)}"`);
  const cls = (name) => (offered[name].available ? '' : 'off');
  const boxes = (name, label, values, fmt) => !values ? `<div class="row ${cls(name)}"><span class="lbl">${label}</span><span class="note">${esc(offered[name].reason)}</span></div>` :
    `<div class="row"><span class="lbl">${label}</span>` + values.map((v) =>
      `<label><input type="checkbox" data-f="${name}" value="${v}" ${!(t[name] || []).length || t[name].includes(v) ? 'checked' : ''}> ${esc(fmt(v))}</label>`).join('') + '</div>';
  $('trialFilters').innerHTML =
    `<label class="check ${cls('bwm_include')}"><input type="checkbox" data-f="bwm_include" ${t.bwm_include ? 'checked' : ''} ${off('bwm_include')}> BWM trial inclusion only</label>` +
    `<label class="check ${cls('exclude_nogo')}"><input type="checkbox" data-f="exclude_nogo" ${t.exclude_nogo ? 'checked' : ''} ${off('exclude_nogo')}> Exclude no-go</label>` +
    boxes('contrasts', 'Contrasts', lv.contrasts, (v) => `${v * 100}%`) +
    boxes('blocks', 'Blocks', lv.blocks, (v) => `${v}`) +
    boxes('outcomes', 'Outcomes', lv.outcomes, (v) => (v < 0 ? 'error' : 'reward'));
}
function readTrialFilters() {
  const t = {}, box = $('trialFilters');
  for (const name of ['bwm_include', 'exclude_nogo']) {
    const i = box.querySelector(`input[data-f="${name}"]`);
    t[name] = Boolean(i && i.checked && !i.disabled);
  }
  for (const name of ['contrasts', 'blocks', 'outcomes']) {
    const all = [...box.querySelectorAll(`input[data-f="${name}"]`)];
    const on = all.filter((i) => i.checked).map((i) => +i.value);
    t[name] = on.length === all.length ? [] : on;  // all ticked means no filter
  }
  return t;
}
function renderTrialCounts(c) {
  if (!c) return;
  const why = Object.entries(c.excluded).map(([r, n]) => `${n} ${r}`).join(', ');
  $('trialCounts').textContent = c.n_excluded
    ? `${c.n_kept} of ${c.n_total} trials kept · excluded: ${why}`
    : `All ${c.n_total} trials kept.`;
}

// ---------- rail: selectivity ----------
function renderSelectivity(t) {
  const s = state.session, pair = s.comparisons[state.split];
  $('runSel').disabled = !pair;
  if (!state.split) {
    $('selWhat').textContent = 'Choose a condition to split by.';
  } else if (!pair) {
    $('selWhat').textContent = `${s.conditions[state.split]} has many levels: read its tuning curve. No two-condition test.`;
  } else {
    $('selWhat').textContent = `AUROC ${pair[1]} vs ${pair[0]} per unit: above 0.5, a higher rate for ${pair[1]}. ` +
      `Tested against a null, Benjamini–Hochberg across the units tested, α = ${s.response.alpha}.`;
  }
  if (!t) {
    $('selSummary').textContent = pair ? 'Not run for this event and split.' : '';
    return;
  }
  const on = t.probes.length === 1 ? `probe ${t.probes[0]}` : `probes ${t.probes.join(', ')}`;
  $('selSummary').textContent = `${t.n_selective} of ${t.n_tests} units on ${on} selective ` +
    `(${t.n_higher_b} higher for ${t.b}, ${t.n_higher_a} for ${t.a}) · ${t.n_a} vs ${t.n_b} trials · ` +
    `null: ${t.null}, ${t.n_null.toLocaleString()} draws, seed ${t.seed} · ${t.window} window`;
}
async function runSelectivity() {
  const button = $('runSel');
  button.disabled = true;
  $('selSummary').textContent = 'Testing…';
  try {
    await getJSON('/api/selectivity?' + params({ responsive: 0 }));
    await loadUnits();
  } catch (e) {
    $('selSummary').textContent = e.message;
  } finally {
    button.disabled = !state.session.comparisons[state.split];
  }
}

// ---------- rail: region tree ----------
function renderTree() {
  const el = $('tree');
  if (!state.tree.length) {
    el.innerHTML = `<p class="note">${esc(state.session.missing['units.acronym'] || 'No regions.')}</p>`;
    return;
  }
  const items = [`<div class="node" data-node="" aria-selected="${state.node === ''}"><span>All regions</span><span class="n">${state.tree[0].n_total}</span></div>`];
  for (const n of state.tree) {
    const label = n.acronym === 'root' && n.n_direct
      ? `root <span class="missing">· ${n.n_direct} in no ${esc(state.level)} region</span>`
      : esc(n.acronym);
    items.push(
      `<div class="node" data-node="${esc(n.acronym)}" aria-selected="${state.node === n.acronym}" style="padding-left:${6 + n.depth * 10}px" title="${esc(n.name)}">` +
        `<span class="sw" style="background:${n.colour}"></span><span>${label}</span><span class="n">${n.n_total}</span></div>`,
    );
  }
  el.innerHTML = items.join('');
}

// ---------- unit table ----------
function renderTable() {
  const names = Object.fromEntries(state.tree.map((n) => [n.acronym, n.name]));
  $('tableMeta').textContent = `${state.rows.length} shown${state.node ? ` · in ${state.node}` : ''}`;
  $('rows').innerHTML = state.rows
    .map((u) => {
      const region = u.region_level == null
        ? fmt(null)
        : `<span class="region" title="${esc(names[u.region_level] || '')} · Allen: ${esc(u.region)}"><span class="sw" style="background:${u.colour}"></span>${esc(u.region_level)}</span>`;
      return `<tr data-id="${esc(u.id)}" aria-selected="${u.id === state.unit}"><td>${esc(u.id)}</td><td>${region}</td>` +
        `<td class="num">${fmt(u.depth_um, 0)}</td><td class="num">${fmt(u.firing_rate_hz, 2)}</td><td>${fmt(u.label, 2)}</td>` +
        `<td class="${u.qc_passed ? '' : 'fail'}" title="${esc(u.qc_reason)}">${u.qc_passed ? 'pass' : 'fail'}</td>` +
        `<td class="resp" title="${respTitle(u)}">${{ up: '↑', down: '↓', no: '·' }[u.resp] || fmt(null)}</td>` +
        `<td class="num" title="${selTitle(u)}">${u.sel_auroc == null ? fmt(null) : u.sel_auroc.toFixed(2) + (u.sel ? '*' : '')}</td>` +
        `<td class="resp" title="${lockTitle(u)}">${u.locked == null ? fmt(null) : u.locked ? (u.lock_hz > 0 ? '↑' : '↓') : '·'}</td></tr>`;
    })
    .join('');
}

function selTitle(u) {
  if (u.sel_auroc == null) return 'Not tested';
  return `AUROC ${u.sel_auroc.toFixed(3)}, p = ${u.sel_p.toPrecision(2)}, q = ${u.sel_q.toPrecision(2)}`;
}
function lockTitle(u) {
  if (u.locked == null) return 'Not tested';
  return `Δ at movement = ${u.lock_hz.toFixed(2)} Hz, p = ${u.lock_p.toPrecision(2)}, q = ${u.lock_q.toPrecision(2)}`;
}
function respTitle(u) {
  if (!u.resp) return 'Not tested';
  return `Δ = ${u.resp_hz.toFixed(2)} Hz, p = ${u.resp_p.toPrecision(2)}, q = ${u.resp_q.toPrecision(2)}`;
}

function selectUnit(id, { scroll = true } = {}) {
  state.unit = id;
  for (const tr of $('rows').querySelectorAll('tr')) tr.setAttribute('aria-selected', tr.dataset.id === id);
  if (scroll && id) $('rows').querySelector(`tr[data-id="${CSS.escape(id)}"]`)?.scrollIntoView({ block: 'nearest' });
  plotUnit();
  renderProbe();
  brain.select(id);
}

// ---------- plots (PNGs from viz/) ----------
const latest = { unit: 0, pop: 0, tuning: 0, wheel: 0 };
async function fetchPlot(kind, url, img, caption, err) {
  const seq = ++latest[kind];
  const r = await fetch(url);
  if (seq !== latest[kind]) return null;  // a newer request replaced this one
  if (!r.ok) {
    err.textContent = await r.text();
    img.style.opacity = 0.25;
    return null;
  }
  err.textContent = '';
  img.style.opacity = 1;
  caption.textContent = decodeURIComponent(r.headers.get('X-Caption') || '');
  img.src = URL.createObjectURL(await r.blob());
  return r.headers;
}
function plotUnit() {
  if (state.unit) fetchPlot('unit', '/api/unit.png?' + params({ unit: state.unit }), $('unitImg'), $('unitCaption'), $('unitErr'));
  plotWheel();
  plotTuning();
}
// The wheel is the same for every unit: fetch it only when the event, window, bins or
// trials change. Without a wheel, say why instead of drawing an empty panel.
let wheelUrl = '';
function plotWheel() {
  const m = state.session.movement;
  $('wheelImg').hidden = !m.wheel;
  if (!m.wheel) {
    $('wheelCaption').textContent = `No wheel: ${m.wheel_missing || 'not in this session'}.`;
    return;
  }
  const url = '/api/wheel.png?' + params({ unit: '', node: '', probe: '', all: 0, responsive: 0, split: '' });
  if (url === wheelUrl) return;
  wheelUrl = url;
  fetchPlot('wheel', url, $('wheelImg'), $('wheelCaption'), $('wheelErr'));
}
function plotTuning() {
  $('tuningBox').hidden = !state.split || !state.unit;
  if (!$('tuningBox').hidden) {
    fetchPlot('tuning', '/api/tuning.png?' + params({ unit: state.unit }), $('tuningImg'), $('tuningCaption'), $('tuningErr'));
  }
}
async function plotPop() {
  const h = await fetchPlot('pop', '/api/population.png?' + params(), $('popImg'), $('popCaption'), $('popErr'));
  if (h) {
    state.popRows = h.get('X-Rows').split(',');
    state.box = h.get('X-Box').split(',').map(Number);
  }
}
function popRowAt(e) {
  const b = state.box;
  if (!b || !state.popRows) return null;
  const rect = $('popImg').getBoundingClientRect();
  const fx = (e.clientX - rect.left) / rect.width;
  const fy = (e.clientY - rect.top) / rect.height;
  if (fx < b[0] || fx > b[2] || fy < b[1] || fy >= b[3]) return null;
  const i = Math.floor(((fy - b[1]) / (b[3] - b[1])) * state.popRows.length);
  return { id: state.popRows[i], x: e.clientX - rect.left, y: e.clientY - rect.top };
}

// ---------- probe strip ----------
let probeSeq = 0;
async function renderProbe() {
  const svg = $('probe');
  if (!state.unit) { svg.innerHTML = ''; return; }
  const seq = ++probeSeq;
  const d = await getJSON('/api/probe?' + params({ unit: state.unit }));
  if (seq !== probeSeq) return;
  const W = svg.clientWidth || 150, H = svg.clientHeight || 420, pad = 14;
  const depths = d.units.map((u) => u.depth_um).filter((v) => v != null);
  if (!depths.length) {
    const why = state.session.missing['units.depths'] || 'no unit on this probe has a depth';
    svg.innerHTML = `<foreignObject x="0" y="0" width="${W}" height="${H}"><p class="note" xmlns="http://www.w3.org/1999/xhtml">No probe strip: ${esc(why)}</p></foreignObject>`;
    $('brainMeta').textContent = '';
    return;
  }
  const top = Math.ceil((Math.max(...depths) + 100) / 500) * 500;
  const y = (um) => H - pad - (um / top) * (H - 2 * pad);  // tip at the bottom
  const shank = { x: 46, w: 30 }, lateralMax = 70;
  const parts = [`<rect class="shank" x="${shank.x}" y="${pad}" width="${shank.w}" height="${H - 2 * pad}" rx="4"/>`];
  for (let um = 0; um <= top; um += 500) parts.push(`<text class="tick" x="28" y="${y(um) + 3}" text-anchor="end">${um}</text>`);
  for (const r of d.runs) {
    const y0 = y(r.bottom_um + 10), y1 = y(r.top_um - 10);
    parts.push(`<rect x="34" y="${y0}" width="8" height="${Math.max(2, y1 - y0)}" rx="2" fill="${visible(r.colour)}"><title>${esc(r.region)} · ${r.n_units} units</title></rect>`);
    if (y1 - y0 >= 11) parts.push(`<text x="${shank.x + shank.w + 6}" y="${(y0 + y1) / 2 + 3}">${esc(r.region)}</text>`);
  }
  for (const u of d.units) {
    if (u.depth_um == null) continue;
    const lx = u.lateral_um == null ? shank.w / 2 : (u.lateral_um / lateralMax) * shank.w;
    const sel = u.id === state.unit;
    parts.push(`<circle data-id="${esc(u.id)}" class="${sel ? 'sel' : ''}" cx="${shank.x + lx}" cy="${y(u.depth_um)}" r="${sel ? 5 : 3.5}" fill="${visible(u.colour)}"><title>${esc(u.id)}${u.region ? ' · ' + esc(u.region) : ''}</title></circle>`);
  }
  svg.innerHTML = parts.join('');
  svg.querySelector('circle.sel')?.parentNode.appendChild(svg.querySelector('circle.sel'));  // draw on top
  $('brainMeta').textContent = `${d.probe} · depth from tip (µm) · region bars span recorded units only`;
}

// ---------- 3D brain (three.js) ----------
const brain = (() => {
  const el = $('brain');
  let renderer, scene, camera, controls, points, marker, ids = [], positions = new Map(), framed = false, generation = 0;
  const meshes = new Map(), loader = new OBJLoader(), regionGroup = new THREE.Group(), trackGroup = new THREE.Group();
  const brainMaterial = new THREE.MeshLambertMaterial({ transparent: true, opacity: 0.1, depthWrite: false });
  const trackMaterial = new THREE.LineBasicMaterial();
  const note = (text) => { $('brainNote').textContent = text; };
  const render = () => renderer && renderer.render(scene, camera);

  function dot() {
    const c = document.createElement('canvas');
    c.width = c.height = 64;
    const g = c.getContext('2d');
    g.beginPath(); g.arc(32, 32, 28, 0, 2 * Math.PI); g.fillStyle = '#fff'; g.fill();
    return new THREE.CanvasTexture(c);
  }

  function init() {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(devicePixelRatio);
    el.prepend(renderer.domElement);
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(35, 1, 10, 200000);
    camera.up.set(0, -1, 0);  // CCF dorsoventral grows ventrally, so dorsal is up
    controls = new OrbitControls(camera, renderer.domElement);
    controls.addEventListener('change', render);
    const light = new THREE.DirectionalLight(0xffffff, 1.6);
    light.position.set(-0.5, -1, 0.6);
    camera.add(light);
    scene.add(camera, new THREE.AmbientLight(0xffffff, 1.1), regionGroup, trackGroup);
    marker = new THREE.Mesh(new THREE.SphereGeometry(150, 24, 16), new THREE.MeshBasicMaterial());
    marker.visible = false;
    scene.add(marker);
    new ResizeObserver(() => {
      const w = el.clientWidth, h = el.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      render();
    }).observe(el);
    renderer.domElement.addEventListener('click', pick);
    applyTheme();
  }

  function applyTheme() {
    brainMaterial.color.set(css('--muted'));
    trackMaterial.color.set(css('--ink2'));
    marker.material.color.set(css('--ink'));
    render();
  }

  function frame(box) {
    const centre = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3()).length();
    controls.target.copy(centre);
    camera.position.copy(centre).add(new THREE.Vector3(-0.55 * size, -0.75 * size, 0.95 * size));
    controls.update();
  }

  function mesh(id) {
    if (!meshes.has(id)) meshes.set(id, loader.loadAsync(`/mesh/${id}.obj`));
    return meshes.get(id);
  }

  function pick(e) {
    if (!points) return;
    const rect = renderer.domElement.getBoundingClientRect();
    const ray = new THREE.Raycaster();
    ray.params.Points.threshold = 100;
    ray.setFromCamera(new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1), camera);
    const hit = ray.intersectObject(points)[0];
    if (hit) selectUnit(ids[hit.index]);
  }

  async function load(g) {
    if (!renderer) init();
    const gen = ++generation;
    if (g.missing) {
      note(`No 3D view: ${g.missing}`);
      return;
    }
    if (points) scene.remove(points);
    trackGroup.clear();
    ids = g.units.map((u) => u.id);
    positions = new Map(g.units.map((u) => [u.id, u.ccf]));
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(g.units.flatMap((u) => u.ccf), 3));
    geom.setAttribute('color', new THREE.Float32BufferAttribute(g.units.flatMap((u) => new THREE.Color(visible(u.colour)).toArray()), 3));
    points = new THREE.Points(geom, new THREE.PointsMaterial({ size: 160, vertexColors: true, map: dot(), alphaTest: 0.5 }));
    scene.add(points);
    for (const t of g.tracks) {
      trackGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...t.a), new THREE.Vector3(...t.b)]), trackMaterial));
    }
    if (!framed) { geom.computeBoundingBox(); frame(geom.boundingBox.clone().expandByScalar(2500)); }
    render();
    note('Loading meshes… the first view downloads them from the Allen Institute.');
    try {
      const whole = await mesh(g.brain_id);
      if (gen !== generation) return;
      whole.traverse((o) => { if (o.isMesh) o.material = brainMaterial; });
      scene.add(whole);
      if (!framed) { frame(new THREE.Box3().setFromObject(whole)); framed = true; }
      const regions = await Promise.all(g.meshes.map(async (m) => {
        const obj = await mesh(m.id);
        obj.traverse((o) => { if (o.isMesh) o.material = new THREE.MeshLambertMaterial({ color: m.colour, transparent: true, opacity: 0.3, depthWrite: false }); });
        obj.userData.acronym = m.acronym;
        return obj;
      }));
      if (gen !== generation) return;
      regionGroup.clear();
      regionGroup.add(...regions);
      note(`${g.meshes.length} ${state.level} regions · ${ids.length} units · ${g.tracks.length} probe track${g.tracks.length === 1 ? '' : 's'} · drag to rotate, click a unit to select it`);
    } catch (e) {
      note(`Meshes unavailable: ${e.message}`);
    }
    render();
  }

  function select(id) {
    if (!marker) return;
    const p = positions.get(id);
    marker.visible = Boolean(p);
    if (p) marker.position.set(...p);
    render();
  }

  return { load, select, note, theme: () => renderer && applyTheme() };
})();

// ---------- wiring ----------
$('theme').addEventListener('click', (e) => { if (e.target.dataset.v) setTheme(e.target.dataset.v); });
$('level').addEventListener('click', (e) => {
  const v = e.target.dataset.v;
  if (!v || e.target.disabled || v === state.level) return;
  state.level = v;
  state.node = '';  // a node at one level need not exist at another
  for (const b of $('level').querySelectorAll('button')) b.setAttribute('aria-pressed', b.dataset.v === v);
  loadUnits();
  loadGeometry();
});
$('all').addEventListener('change', () => { state.all = $('all').checked; resetResponsive(); loadUnits(); loadGeometry(); });
$('runTest').addEventListener('click', runTest);
$('runSel').addEventListener('click', runSelectivity);
$('runLock').addEventListener('click', runLocking);
$('movementFree').addEventListener('change', () => {
  state.movementFree = $('movementFree').checked;
  resetResponsive();  // results on all trials and on movement-free trials are kept apart
  loadUnits();
});
$('split').addEventListener('change', () => { state.split = $('split').value; loadUnits(); });
$('trialFilters').addEventListener('change', () => {
  state.trials = readTrialFilters();
  resetResponsive();  // test results belong to the trials they used
  loadUnits();
});
$('probeFilter').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b || b.dataset.v === state.probe) return;
  state.probe = b.dataset.v;
  resetResponsive();  // a test belongs to the units it corrected over
  renderProbes();
  loadUnits();
  loadGeometry();
});
$('save').addEventListener('click', () => act($('save'), 'Saving…', async () => {
  const d = await post('/api/project', currentView());
  $('projectStatus').title = d.path;
  return `Saved ${d.path.split('/').pop()} at ${new Date().toLocaleTimeString()}`;
}));
$('export').addEventListener('click', () => act($('export'), 'Exporting…', async () => {
  const d = await post('/api/export', currentView());
  $('projectStatus').title = d.folder;
  return `Exported ${d.files.length} files to runs/${d.folder.split('/').pop()}`;
}));
$('responsive').addEventListener('change', () => { state.responsive = $('responsive').checked; loadUnits(); });
$('event').addEventListener('change', () => {
  resetResponsive();
  renderMovementNotes();
  if ($('movementFree').disabled) state.movementFree = $('movementFree').checked = false;
  loadUnits();
});
$('tree').addEventListener('click', (e) => {
  const n = e.target.closest('.node');
  if (!n) return;
  state.node = n.dataset.node;
  loadUnits();
});
$('rows').addEventListener('click', (e) => { const tr = e.target.closest('tr'); if (tr) selectUnit(tr.dataset.id); });
$('probe').addEventListener('click', (e) => { const c = e.target.closest('circle'); if (c) selectUnit(c.dataset.id); });
for (const id of ['t0', 't1', 'bin', 'baseline', 'b0', 'b1']) $(id).addEventListener('change', () => { plotUnit(); plotPop(); });
$('popImg').addEventListener('mousemove', (e) => {
  const hit = popRowAt(e), tip = $('popTip');
  tip.hidden = !hit;
  if (!hit) return;
  const row = state.rows.find((u) => u.id === hit.id);
  tip.textContent = row?.region_level ? `${hit.id} · ${row.region_level}` : hit.id;
  tip.style.left = `${hit.x}px`;
  tip.style.top = `${hit.y}px`;
});
$('popImg').addEventListener('mouseleave', () => { $('popTip').hidden = true; });
$('popImg').addEventListener('click', (e) => { const hit = popRowAt(e); if (hit) selectUnit(hit.id); });

try { const t = localStorage.getItem('studio-theme'); if (t && t !== 'auto') setTheme(t); } catch { /* ignore */ }
init().catch((e) => { $('source').textContent = e.message; });
