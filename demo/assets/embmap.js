/* Linked 2-D maps of real query embeddings (data: window.EMB from build_embeddings.py). */
(function () {
  "use strict";
  const E = window.EMB;
  const root = document.getElementById("embMap");
  if (!E || !root) return;
  const NS = "http://www.w3.org/2000/svg";
  const cv = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  const PAL = ["#2f6db3", "#d1631f", "#0f8a7e", "#8e5aa8", "#c9a227", "#c23b5a", "#4a9b3f", "#6b7fd6"];
  const NCOL = PAL.length;

  let g = "dbpedia", sel = null, hover = null;
  const panels = [
    { svg: document.getElementById("embL"), pick: document.getElementById("pickL"), stat: document.getElementById("statL"), key: null },
    { svg: document.getElementById("embR"), pick: document.getElementById("pickR"), stat: document.getElementById("statR"), key: null },
  ];
  const tip = document.getElementById("embTip");
  const info = document.getElementById("embInfo");

  function defaults() {
    const sp = E[g].spaces;
    if (sp.zeroshot && sp.finetuned) return ["zeroshot", "finetuned"];
    return ["tfidf", sp.finetuned ? "finetuned" : "target"];
  }
  function fillPickers() {
    const sp = E[g].spaces;
    const d = defaults();
    panels.forEach((p, k) => {
      p.pick.innerHTML = Object.keys(sp).map((key) => `<option value="${key}">${esc(sp[key].label)}</option>`).join("");
      if (!p.key || !sp[p.key]) p.key = d[k];
      p.pick.value = p.key;
    });
    const note = document.getElementById("embNote");
    if (sp.finetuned) note.hidden = true;
    else note.hidden = false;
  }
  function colorOf(q) { return q.g >= 0 && q.g < NCOL ? PAL[q.g] : cv("--mid"); }

  function draw() {
    const D = E[g];
    const qs = D.queries;
    const focus = sel ?? hover;
    const posSet = new Set(focus != null ? (D.pos[String(focus)] || []) : []);
    panels.forEach((p) => {
      const S = D.spaces[p.key];
      const svg = p.svg;
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const W = 400, H = 400, pad = 14;
      const X = (v) => pad + (v + 1) / 2 * (W - 2 * pad), Y = (v) => pad + (1 - (v + 1) / 2) * (H - 2 * pad);
      el("rect", { x: 0, y: 0, width: W, height: H, rx: 12, fill: cv("--surface-2") }, svg);
      const dim = focus != null;
      // points: non-highlighted first
      const order = qs.map((_, i) => i).sort((a, b) => (qs[a].g >= 0 && qs[a].g < NCOL) - (qs[b].g >= 0 && qs[b].g < NCOL));
      for (const i of order) {
        const [x, y] = S.xy[i];
        const on = !dim || i === focus || posSet.has(i) || i === S.nn[focus];
        const c = el("circle", {
          cx: X(x), cy: Y(y), r: on && dim ? 5 : 3.6,
          fill: colorOf(qs[i]), opacity: on ? (qs[i].g >= 0 && qs[i].g < NCOL ? 0.95 : 0.55) : 0.2,
          "data-i": i,
        }, svg);
        c.style.cursor = "pointer";
      }
      if (focus != null) {
        const [fx, fy] = S.xy[focus];
        const nb = S.nn[focus];
        const ok = posSet.has(nb);
        const [nx, ny] = S.xy[nb];
        el("line", { x1: X(fx), y1: Y(fy), x2: X(nx), y2: Y(ny), stroke: ok ? cv("--good") : cv("--bad"), "stroke-width": 2.5 }, svg);
        for (const j of posSet) {
          const [px, py] = S.xy[j];
          el("circle", { cx: X(px), cy: Y(py), r: 8, fill: "none", stroke: cv("--good"), "stroke-width": 1.6 }, svg);
        }
        el("circle", { cx: X(nx), cy: Y(ny), r: 9, fill: "none", stroke: ok ? cv("--good") : cv("--bad"), "stroke-width": 2.2 }, svg);
        el("circle", { cx: X(fx), cy: Y(fy), r: 8, fill: cv("--ink"), stroke: cv("--surface"), "stroke-width": 2 }, svg);
      }
      p.stat.innerHTML = `Nearest neighbour returns (nearly) the same answers for <b>${Math.round(S.hits1 * 100)}%</b> of queries`;
      p.stat.className = "embstat " + (S.kind === "ours" || S.kind === "target" ? "good" : "");
    });
    renderInfo();
  }

  function renderInfo() {
    const D = E[g];
    const f = sel ?? hover;
    if (f == null) {
      info.innerHTML = `<p class="note">Hover or click a dot. Its <b style="color:var(--good)">ringed</b> partners return (nearly) the same answers (J ≥ 0.8). The line goes to its nearest neighbour in each map: <b style="color:var(--good)">green</b> if that neighbour shares the answers, <b style="color:var(--bad)">red</b> if not.</p>`;
      return;
    }
    const q = D.queries[f];
    const pos = new Set(D.pos[String(f)] || []);
    const rows = panels.map((p) => {
      const S = D.spaces[p.key], nb = S.nn[f], ok = pos.has(nb);
      return `<div class="nnrow"><span class="nnlab">${esc(S.label)}</span><code>${esc(D.queries[nb].q)}</code><span class="jb ${ok ? "pos" : "neg"}">${ok ? "same answers ✓" : "different answers ✗"}</span></div>`;
    }).join("");
    info.innerHTML = `<div class="selq"><span class="tag ta">query</span><code>${esc(q.q)}</code><small>${(q.n || 0).toLocaleString()} answers · ${pos.size} quer${pos.size === 1 ? "y" : "ies"} with the same answers</small></div><h4>Its nearest neighbour</h4>${rows}${sel != null ? '<button id="embClear" class="linkbtn">clear selection</button>' : ""}`;
    const b = document.getElementById("embClear");
    if (b) b.onclick = () => { sel = null; draw(); };
  }

  function idxFromEvent(ev) {
    const t = ev.target;
    return t && t.dataset && t.dataset.i != null ? +t.dataset.i : null;
  }
  panels.forEach((p, k) => {
    p.pick.addEventListener("change", () => { p.key = p.pick.value; draw(); });
    p.svg.addEventListener("mousemove", (ev) => {
      const i = idxFromEvent(ev);
      if (i == null) { tip.hidden = true; if (hover != null) { hover = null; if (sel == null) draw(); } return; }
      tip.hidden = false;
      tip.textContent = E[g].queries[i].q;
      const r = root.getBoundingClientRect();
      tip.style.left = Math.min(ev.clientX - r.left + 12, r.width - 300) + "px";
      tip.style.top = ev.clientY - r.top + 14 + "px";
      if (hover !== i) { hover = i; if (sel == null) draw(); }
    });
    p.svg.addEventListener("mouseleave", () => { tip.hidden = true; hover = null; if (sel == null) draw(); });
    p.svg.addEventListener("click", (ev) => { const i = idxFromEvent(ev); sel = i; draw(); });
  });
  document.getElementById("embSeg").addEventListener("click", (ev) => {
    const b = ev.target.closest("button"); if (!b) return;
    document.querySelectorAll("#embSeg button").forEach((x) => x.classList.toggle("on", x === b));
    g = b.dataset.g; sel = null; hover = null; panels.forEach((p) => (p.key = null)); fillPickers(); draw();
  });
  document.getElementById("embRandom").addEventListener("click", () => {
    // pick a query that has partners and where the two maps disagree, to show the difference
    const D = E[g], L = D.spaces[panels[0].key], R = D.spaces[panels[1].key];
    const cand = Object.keys(D.pos).map(Number).filter((i) => {
      const p = new Set(D.pos[String(i)]);
      return p.has(R.nn[i]) !== p.has(L.nn[i]);
    });
    const pool = cand.length ? cand : Object.keys(D.pos).map(Number);
    sel = pool[Math.floor(Math.random() * pool.length)];
    draw();
  });
  fillPickers();
  draw();
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
})();
