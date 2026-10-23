"use strict";

const ROUNDS_PER_GAME = 5;
const EUROPE = [[35, -11], [70, 45]];
const BANDS = [
  { max: 50, label: "< 50 km" },
  { max: 200, label: "50–200 km" },
  { max: 750, label: "200–750 km" },
  { max: Infinity, label: "> 750 km" },
];
const SVGNS = "http://www.w3.org/2000/svg";

const $ = (s) => document.querySelector(s);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const prettyCountry = (c) => c.replace(/_/g, " ");
const fmtKm = (d) => (d < 10 ? d.toFixed(1) : Math.round(d).toLocaleString("en")) + " km";
const points = (d) => Math.round(5000 * Math.exp(-d / 500));
const bandOf = (d) => BANDS.findIndex((b) => d < b.max);
const darkQuery = matchMedia("(prefers-color-scheme: dark)");

function isDark() {
  const t = document.documentElement.dataset.theme;
  return t ? t === "dark" : darkQuery.matches;
}

function haversine(lat1, lng1, lat2, lng2) {
  const r = Math.PI / 180, R = 6371.0088;
  const a = Math.sin(((lat2 - lat1) * r) / 2) ** 2 +
    Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin(((lng2 - lng1) * r) / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(Math.min(1, Math.max(0, a))));
}

function median(arr) {
  const s = [...arr].sort((a, b) => a - b), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}

/* ---------- tiles (follow light/dark) ---------- */
const tileLayers = [];
function addTiles(map) {
  const esri = (name) => `https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/${name}/MapServer/tile/{z}/{y}/{x}`;
  const make = () => {
    const shade = isDark() ? "Dark" : "Light";
    return L.layerGroup([
      L.tileLayer(esri(`World_${shade}_Gray_Base`), { maxZoom: 16,
        attribution: "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors" }),
      L.tileLayer(esri(`World_${shade}_Gray_Reference`), { maxZoom: 16 }),
    ]);
  };
  const entry = { map, layer: make().addTo(map), make };
  tileLayers.push(entry);
}
function refreshTheme() {
  for (const t of tileLayers) { t.map.removeLayer(t.layer); t.layer = t.make().addTo(t.map); }
  if (explore.ready) { restyleDots(); renderCountryChart(); renderHist(); }
  if (lbData) renderLeaderboard(lbData);
}
darkQuery.addEventListener("change", refreshTheme);

/* ---------- tooltip ---------- */
const tip = $("#tooltip");
function showTip(evt, html) {
  tip.innerHTML = html;
  tip.hidden = false;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > innerWidth - 8) x = evt.clientX - w - pad;
  if (y + h > innerHeight - 8) y = evt.clientY - h - pad;
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}
const hideTip = () => { tip.hidden = true; };

/* ---------- tabs ---------- */
const tabInit = {};
document.querySelectorAll(".tabs button").forEach((btn) => {
  btn.addEventListener("click", () => selectTab(btn.dataset.tab));
});
function selectTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((p) => (p.hidden = p.id !== "tab-" + name));
  if (tabInit[name]) tabInit[name]();
  if (name === "play") play.map.invalidateSize();
  if (name === "explore" && explore.map) explore.map.invalidateSize();
  history.replaceState(null, "", "#" + name);
}

/* ---------- PLAY ---------- */
const play = { map: null, rounds: [], deck: [], i: 0, guess: null, layers: [], log: [], done: false };

function pinIcon(kind) {
  return L.divIcon({ className: "", html: `<div class="pin ${kind}"></div>`, iconSize: [18, 18], iconAnchor: [9, 9] });
}

function initPlay(rounds) {
  play.rounds = rounds;
  play.map = L.map("play-map", { worldCopyJump: true, zoomSnap: 0.5 }).fitBounds(EUROPE);
  addTiles(play.map);
  play.map.on("click", (e) => {
    if (play.done) return;
    play.guess = e.latlng;
    if (play.guessMarker) play.guessMarker.setLatLng(e.latlng);
    else play.guessMarker = L.marker(e.latlng, { icon: pinIcon("you"), keyboard: false }).addTo(play.map);
    $("#guess-btn").disabled = false;
    $("#play-hint").textContent = "Happy with it? Hit Guess (or press Enter).";
  });
  $("#guess-btn").addEventListener("click", onButton);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !$("#tab-play").hidden && !$("#guess-btn").disabled &&
        !["BUTTON", "SELECT", "A"].includes(document.activeElement.tagName)) onButton();
  });
  newGame();
}

function newGame() {
  play.deck = shuffle(play.rounds).slice(0, ROUNDS_PER_GAME);
  play.i = 0;
  play.log = [];
  $("#final").hidden = true;
  updateScore();
  startRound();
}

function startRound() {
  const r = play.deck[play.i];
  play.done = false;
  play.guess = null;
  play.layers.forEach((l) => play.map.removeLayer(l));
  play.layers = [];
  if (play.guessMarker) { play.map.removeLayer(play.guessMarker); play.guessMarker = null; }
  play.map.fitBounds(EUROPE);
  $("#photo").src = "img/" + r.img;
  $("#round-label").textContent = `Round ${play.i + 1} / ${ROUNDS_PER_GAME}`;
  $("#result").hidden = true;
  $("#guess-btn").textContent = "Guess";
  $("#guess-btn").disabled = true;
  $("#play-hint").textContent = "Click the map to drop your guess.";
  const next = play.deck[play.i + 1];
  if (next) new Image().src = "img/" + next.img;
}

function onButton() {
  if (!play.done) return reveal();
  if (play.i + 1 < ROUNDS_PER_GAME) { play.i++; startRound(); }
  else if (!$("#final").hidden) newGame();
  else showFinal();
}

function reveal() {
  if (!play.guess) return;
  const r = play.deck[play.i];
  play.done = true;
  const truth = [r.lat, r.lng], model = [r.pred_lat, r.pred_lng];
  const you = [play.guess.lat, play.guess.lng];
  const dYou = haversine(...you, ...truth), dModel = r.err;
  const pYou = points(dYou), pModel = points(dModel);
  play.log.push({ country: r.country, dYou, dModel, pYou, pModel });

  const line = (a, color) => L.polyline([a, truth], { color, weight: 2, dashArray: "5 6", opacity: 0.9 }).addTo(play.map);
  play.layers.push(
    line(you, css("--you")),
    line(model, css("--model")),
    L.marker(model, { icon: pinIcon("model") }).bindTooltip("Model").addTo(play.map),
    L.marker(truth, { icon: pinIcon("truth"), zIndexOffset: 1000 }).bindTooltip("True location").addTo(play.map),
  );
  play.map.fitBounds(L.latLngBounds([truth, you, model]).pad(0.25), { maxZoom: 8 });

  const verdict = Math.abs(dYou - dModel) < 1 ? "A tie." : dYou < dModel ? "You beat the model!" : "The model wins this one.";
  const res = $("#result");
  res.innerHTML = `
    <div class="result-row">
      <div><div class="who"><span class="key you"><i></i></span>You</div><div class="big">${fmtKm(dYou)}</div><div class="muted">${pYou.toLocaleString("en")} pts</div></div>
      <div><div class="who"><span class="key model"><i></i></span>Model</div><div class="big">${fmtKm(dModel)}</div><div class="muted">${pModel.toLocaleString("en")} pts</div></div>
      <div><div class="who"><span class="key"><span class="ring"></span></span>Answer</div><div class="big">${prettyCountry(r.country)}</div><div class="muted">${r.lat.toFixed(3)}, ${r.lng.toFixed(3)}</div></div>
      <div class="verdict">${verdict}</div>
    </div>`;
  res.hidden = false;
  updateScore();
  if (innerWidth <= 820) res.scrollIntoView({ behavior: "smooth", block: "nearest" });
  $("#guess-btn").textContent = play.i + 1 < ROUNDS_PER_GAME ? "Next round" : "See results";
  $("#play-hint").textContent = "";
}

function updateScore() {
  const sum = (k) => play.log.reduce((s, r) => s + r[k], 0);
  $("#score-you").textContent = sum("pYou").toLocaleString("en");
  $("#score-model").textContent = sum("pModel").toLocaleString("en");
}

function showFinal() {
  const sum = (k) => play.log.reduce((s, r) => s + r[k], 0);
  const you = sum("pYou"), model = sum("pModel");
  const wins = play.log.filter((r) => r.dYou < r.dModel).length;
  const head = you > model ? "You beat the model 🎉" : you === model ? "Dead heat" : "The model wins";
  const f = $("#final");
  f.innerHTML = `
    <h2>${head}</h2>
    <p class="muted">${you.toLocaleString("en")} vs ${model.toLocaleString("en")} points. You were closer in ${wins} of ${ROUNDS_PER_GAME} rounds.
    Points per round are 5000·e<sup>−d/500 km</sup>.</p>
    <table>
      <thead><tr><th>Round</th><th>Country</th><th>You</th><th>Model</th></tr></thead>
      <tbody>${play.log.map((r, i) => `<tr><td>${i + 1}</td><td>${prettyCountry(r.country)}</td><td>${fmtKm(r.dYou)}</td><td>${fmtKm(r.dModel)}</td></tr>`).join("")}</tbody>
    </table>`;
  f.hidden = false;
  $("#guess-btn").textContent = "Play again";
  f.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ---------- EXPLORE ---------- */
const explore = { ready: false, map: null, rows: [], countries: [], dots: [], filter: "", summary: null, sel: [] };

function bandColor(i) { return css(`--band-${i + 1}`); }

async function initExplore(summary) {
  explore.summary = summary;
  renderStats(summary);
  renderLegend();
  const sel = $("#country-filter");
  summary.countries.map((c) => c.country).sort().forEach((c) => sel.add(new Option(prettyCountry(c), c)));
  sel.addEventListener("change", () => setFilter(sel.value));
  renderTable(summary);

  const pts = await fetch("data/points.json").then((r) => r.json());
  explore.countries = pts.countries;
  explore.rows = pts.rows;
  explore.map = L.map("explore-map", { preferCanvas: true, zoomSnap: 0.5 }).fitBounds(EUROPE);
  addTiles(explore.map);
  const renderer = L.canvas({ padding: 0.3, tolerance: 4 });
  // Draw the worst errors first, so the accurate majority stays visible on top.
  const order = explore.rows.map((_, i) => i).sort((a, b) => explore.rows[b][5] - explore.rows[a][5]);
  explore.dots = new Array(explore.rows.length);
  for (const i of order) {
    const r = explore.rows[i];
    const m = L.circleMarker([r[0], r[1]], { renderer, radius: 3, stroke: false, fillOpacity: 0.85, fillColor: bandColor(bandOf(r[5])) });
    m.on("click", () => selectDot(i));
    m.addTo(explore.map);
    explore.dots[i] = m;
  }
  explore.ready = true;
  renderCountryChart();
  renderHist();
}

function restyleDots() {
  explore.rows.forEach((r, i) => {
    const show = !explore.filter || explore.countries[r[4]] === explore.filter;
    explore.dots[i].setStyle({ fillColor: bandColor(bandOf(r[5])), fillOpacity: show ? 0.85 : 0 });
    if (!show) explore.dots[i].closePopup();
  });
}

function selectDot(i) {
  const r = explore.rows[i];
  explore.sel.forEach((l) => explore.map.removeLayer(l));
  const model = css("--model");
  explore.sel = [
    L.polyline([[r[0], r[1]], [r[2], r[3]]], { color: model, weight: 2, dashArray: "5 6" }).addTo(explore.map),
    L.circleMarker([r[2], r[3]], { radius: 6, color: css("--surface-1"), weight: 2, fillColor: model, fillOpacity: 1 }).addTo(explore.map),
  ];
  L.popup({ autoPan: false })
    .setLatLng([r[0], r[1]])
    .setContent(`<b>${prettyCountry(explore.countries[r[4]])}</b><br>Model was off by <b>${fmtKm(r[5])}</b><br><span style="color:var(--text-muted)">orange dot = model's guess</span>`)
    .openOn(explore.map);
}

function setFilter(c) {
  explore.filter = c;
  $("#country-filter").value = c;
  explore.sel.forEach((l) => explore.map.removeLayer(l));
  explore.sel = [];
  explore.map.closePopup();
  restyleDots();
  if (c) {
    const ll = explore.rows.filter((r) => explore.countries[r[4]] === c).map((r) => [r[0], r[1]]);
    explore.map.fitBounds(L.latLngBounds(ll).pad(0.1));
  } else explore.map.fitBounds(EUROPE);
  renderCountryChart();
  renderHist();
  const s = c ? explore.summary.countries.find((x) => x.country === c) : explore.summary;
  renderStats(s, c);
}

function renderStats(s, c) {
  const items = [
    [fmtKm(s.median), "median error" + (c ? ` · ${prettyCountry(c)}` : "")],
    [fmtKm(s.mean), "mean error"],
    [Math.round(s.lt200 * 100) + "%", "within 200 km"],
    c ? [s.n.toLocaleString("en"), "photos"] : [Math.round(s.lt750 * 100) + "%", "within 750 km"],
  ];
  $("#stats").innerHTML = items.map(([v, l]) => `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
}

function renderLegend() {
  $("#band-legend").innerHTML = "<span>Model error:</span>" +
    BANDS.map((b, i) => `<span><span class="sw" style="background:var(--band-${i + 1})"></span>${b.label}</span>`).join("");
}

function renderTable(summary) {
  $("#country-table").innerHTML =
    `<thead><tr><th>Country</th><th>Photos</th><th>Median</th><th>Mean</th><th>Within 200 km</th></tr></thead><tbody>` +
    summary.countries.map((c) => `<tr><td>${prettyCountry(c.country)}</td><td>${c.n}</td><td>${fmtKm(c.median)}</td><td>${fmtKm(c.mean)}</td><td>${Math.round(c.lt200 * 100)}%</td></tr>`).join("") +
    `<tr><td><b>All</b></td><td>${summary.n.toLocaleString("en")}</td><td><b>${fmtKm(summary.median)}</b></td><td>${fmtKm(summary.mean)}</td><td>${Math.round(summary.lt200 * 100)}%</td></tr></tbody>`;
}

function renderCountryChart() {
  const host = $("#country-chart");
  const data = explore.summary.countries;
  const W = host.clientWidth || 360, rowH = 24, left = 112, right = 64, top = 4;
  const H = top + data.length * rowH;
  const max = Math.max(...data.map((d) => d.median));
  const x = (v) => left + (v / max) * (W - left - right);
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Median error by country" });
  data.forEach((d, i) => {
    const y = top + i * rowH;
    const dim = explore.filter && explore.filter !== d.country;
    const g = el("g", { class: dim ? "dim" : "" }, svg);
    el("text", { x: left - 8, y: y + rowH / 2 + 4, "text-anchor": "end" }, g).textContent = prettyCountry(d.country);
    const w = Math.max(4, x(d.median) - left);
    el("path", { class: "bar", d: roundedRight(left, y + 5, w, rowH - 10, 4), style: "fill:var(--accent)" }, g);
    el("text", { class: "val", x: left + w + 6, y: y + rowH / 2 + 4 }, g).textContent = fmtKm(d.median);
    const hit = el("rect", { class: "hit", x: 0, y, width: W, height: rowH }, svg);
    hit.addEventListener("mousemove", (e) => showTip(e, `<div class="t">${prettyCountry(d.country)}</div>
      <div class="r">Median ${fmtKm(d.median)} · mean ${fmtKm(d.mean)}</div>
      <div class="r">${Math.round(d.lt200 * 100)}% within 200 km · ${d.n} photos</div>`));
    hit.addEventListener("mouseleave", hideTip);
    hit.addEventListener("click", () => setFilter(explore.filter === d.country ? "" : d.country));
  });
  host.replaceChildren(svg);
}

// Bar anchored at the left baseline, rounded only at the data end.
function roundedRight(x, y, w, h, r) {
  r = Math.min(r, w, h / 2);
  return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`;
}
function roundedTop(x, y, w, h, r) {
  r = Math.min(r, w / 2, h);
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}

function renderHist() {
  const host = $("#hist");
  const rows = explore.filter ? explore.rows.filter((r) => explore.countries[r[4]] === explore.filter) : explore.rows;
  const errs = rows.map((r) => r[5]);
  const lo = 0, hi = 3.6, step = 0.1, nb = Math.round((hi - lo) / step);
  const counts = new Array(nb).fill(0);
  for (const e of errs) counts[Math.min(nb - 1, Math.max(0, Math.floor((Math.log10(Math.max(e, 1)) - lo) / step)))]++;
  const W = host.clientWidth || 360, H = 214, l = 36, r = 8, t = 34, b = 28;
  const x = (v) => l + ((v - lo) / (hi - lo)) * (W - l - r);
  const ymax = Math.max(...counts);
  const y = (c) => t + (1 - c / ymax) * (H - t - b);
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Histogram of model error, log scale" });
  const nice = niceStep(ymax);
  for (let v = 0; v <= ymax; v += nice) {
    el("line", { class: "gridline", x1: l, x2: W - r, y1: y(v), y2: y(v) }, svg);
    el("text", { class: "axis", x: l - 6, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = v.toLocaleString("en");
  }
  [1, 10, 100, 1000].forEach((v) => {
    el("text", { class: "axis", x: x(Math.log10(v)), y: H - 8, "text-anchor": "middle" }, svg).textContent = v >= 1000 ? "1,000 km" : v + " km";
  });
  const bw = x(step) - x(0);
  counts.forEach((c, i) => {
    const a = 10 ** (lo + i * step), z = 10 ** (lo + (i + 1) * step);
    const mid = 10 ** (lo + (i + 0.5) * step);
    if (c) el("path", { d: roundedTop(x(lo + i * step) + 1, y(c), bw - 2, H - b - y(c), 2), style: `fill:var(--band-${bandOf(mid) + 1})` }, svg);
    const hit = el("rect", { class: "hit", x: x(lo + i * step), y: t, width: bw, height: H - t - b }, svg);
    hit.addEventListener("mousemove", (e) => showTip(e, `<div class="t">${fmtKm(i === 0 ? 0 : a)} – ${fmtKm(z)}</div><div class="r">${c.toLocaleString("en")} photos (${(100 * c / errs.length).toFixed(1)}%)</div>`));
    hit.addEventListener("mouseleave", hideTip);
  });
  const med = median(errs), mean = errs.reduce((s, v) => s + v, 0) / errs.length;
  [[med, "median"], [mean, "mean"]].forEach(([v, name], k) => {
    const xv = x(Math.log10(Math.max(v, 1)));
    const ly = k ? 26 : 12, right = !k || xv > W - 100;
    el("line", { x1: xv, x2: xv, y1: ly - 8, y2: H - b, style: "stroke:var(--text-primary);stroke-width:1.5", "stroke-dasharray": k ? "3 3" : "" }, svg);
    el("text", { class: "val", x: xv + (right ? -4 : 4), y: ly, "text-anchor": right ? "end" : "start" }, svg).textContent = `${name} ${fmtKm(v)}`;
  });
  host.replaceChildren(svg);
}

function niceStep(max) {
  const raw = max / 4, p = 10 ** Math.floor(Math.log10(raw)), n = raw / p;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p;
}

/* ---------- HOW: leaderboard ---------- */
let lbData = null;
const FAMILIES = [
  { key: "femto", label: "Cells + ConvNeXtV2-femto", color: "--you" },
  { key: "atto", label: "Cells + smaller backbone", color: "--model" },
  { key: "flat", label: "Direct lat/lng regression", color: "--cat-3" },
];
function familyOf(row) {
  if (/flat regression|plain flat-regression/i.test(row.change) || !/cell|femto/i.test(row.name)) return "flat";
  return /femto/i.test(row.backbone) ? "femto" : "atto";
}

function renderLeaderboard(rows) {
  lbData = rows;
  $("#lb-legend").innerHTML = FAMILIES.map((f) => `<span><span class="sw" style="background:var(${f.color})"></span>${f.label}</span>`).join("");
  const host = $("#lb-chart");
  const W = host.clientWidth || 600, H = 280, l = 64, r = 12, t = 14, b = 30;
  const lo = Math.log10(80), hi = Math.log10(1500);
  const x = (rank) => l + ((rank - 1) / (rows.length - 1)) * (W - l - r);
  const y = (km) => t + (1 - (Math.log10(km) - lo) / (hi - lo)) * (H - t - b);
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Median validation error of all 50 experiments, ranked" });
  [100, 200, 500, 1000].forEach((v) => {
    el("line", { class: "gridline", x1: l, x2: W - r, y1: y(v), y2: y(v) }, svg);
    el("text", { class: "axis", x: l - 6, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = v.toLocaleString("en") + " km";
  });
  el("text", { class: "axis", x: l, y: H - 8 }, svg).textContent = "best";
  el("text", { class: "axis", x: W - r, y: H - 8, "text-anchor": "end" }, svg).textContent = "worst (rank 50)";
  rows.forEach((row) => {
    const fam = FAMILIES.find((f) => f.key === familyOf(row));
    const cx = x(row.rank), cy = y(row.km);
    el("circle", { cx, cy, r: 5, style: `fill:var(${fam.color});stroke:var(--surface-1);stroke-width:2` }, svg);
    const hit = el("circle", { class: "hit", cx, cy, r: 10 }, svg);
    hit.addEventListener("mousemove", (e) => showTip(e, `<div class="t">#${row.rank} ${row.name}</div><div class="r">${fmtKm(row.km)} · ${row.backbone}</div><div class="r">${row.change.replace(/`/g, "")}</div>`));
    hit.addEventListener("mouseleave", hideTip);
  });
  const best = rows[0];
  el("text", { class: "val", x: x(1) + 4, y: y(best.km) - 22 }, svg).textContent = `Best: ${best.name}, ${fmtKm(best.km)}`;
  el("line", { x1: x(1) + 2, x2: x(1), y1: y(best.km) - 18, y2: y(best.km) - 7, style: "stroke:var(--text-muted);stroke-width:1" }, svg);
  host.replaceChildren(svg);
}

/* ---------- boot ---------- */
fetch("data/demo.json").then((r) => r.json()).then((data) => {
  initPlay(data.rounds);
  tabInit.explore = () => { if (!explore.summary) initExplore(data.summary); };
  tabInit.how = () => { if (!lbData) renderLeaderboard(data.leaderboard); };
  const start = location.hash.slice(1);
  if (["explore", "how"].includes(start)) selectTab(start);
});

let resizeTimer;
addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (explore.ready && !$("#tab-explore").hidden) { renderCountryChart(); renderHist(); }
    if (lbData && !$("#tab-how").hidden) renderLeaderboard(lbData);
  }, 150);
});
