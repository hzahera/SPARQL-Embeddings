/* Results-Aware SPARQL Embeddings: interactive demo. No dependencies. */
(function () {
  "use strict";
  const P = window.PAPER;
  const D = window.DEMO_DATA;
  const NS = "http://www.w3.org/2000/svg";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const cv = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const fmt = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));
  const pct = (x) => Math.round(x * 100) + "%";
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function s(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs || {}) {
      if (k === "text") e.textContent = attrs[k];
      else e.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(e);
    return e;
  }
  function clear(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }
  function segment(id, cb) {
    const seg = document.getElementById(id);
    seg.addEventListener("click", (ev) => {
      const b = ev.target.closest("button");
      if (!b) return;
      $$("button", seg).forEach((x) => x.classList.toggle("on", x === b));
      cb(b.dataset);
    });
  }
  function rng(seed) {
    let t = seed >>> 0;
    return () => {
      t += 0x6d2b79f5;
      let r = Math.imul(t ^ (t >>> 15), 1 | t);
      r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }
  const redrawers = [];
  const onTheme = (fn) => { redrawers.push(fn); fn(); };
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => redrawers.forEach((f) => f()));

  /* ------------------------------------------------------------------ KPIs */
  (function kpis() {
    if (!$("#kpis")) return;
    const fc = P.retrievalComplete;
    const get = (g, kind) => fc[g].find((r) => r[4] === kind)[1];
    const tf = (g) => fc[g].find((r) => r[0] === "TF-IDF")[1];
    const items = [
      [`10<span class="arrow">/</span>10`, "encoders improve query retrieval after results-aware fine-tuning, on both graphs"],
      [`${fmt(get("dbpedia", "zs"))}<span class="arrow">→</span><span class="to">${fmt(get("dbpedia", "ft"))}</span>`, `DBpedia Hits@1, mean of 10 encoders (TF-IDF: ${fmt(tf("dbpedia"))})`],
      [`${fmt(get("wikidata", "zs"))}<span class="arrow">→</span><span class="to">${fmt(get("wikidata", "ft"))}</span>`, `Wikidata Hits@1, mean of 10 encoders (TF-IDF: ${fmt(tf("wikidata"))})`],
      [`${fmt(Math.max(...P.caching.dbpedia.zsAcc))}<span class="arrow">→</span><span class="to">${fmt(Math.max(...P.caching.dbpedia.ftAcc))}</span>`, "best cache hit accuracy on DBpedia, zero-shot → fine-tuned (θ = 0.8)"],
    ];
    $("#kpis").innerHTML = items.map(([v, t]) => `<div class="kpi"><div class="v">${v}</div><small>${t}</small></div>`).join("");
  })();

  /* ------------------------------------------------------------------ 1. teaser */
  (function teaser() {
    if (!$("#teaserSvg")) return;
    const svg = $("#teaserSvg");
    const layouts = {
      text: { a: [120, 150], b: [168, 122], c: [330, 205] },
      ours: { a: [120, 150], c: [165, 128], b: [325, 60] },
    };
    let mode = "text";
    let cur = JSON.parse(JSON.stringify(layouts.text));
    let anim = null;
    function draw() {
      clear(svg);
      const near = mode === "text" ? "b" : "c";
      const far = near === "b" ? "c" : "b";
      const col = { a: cv("--qa"), b: cv("--qb"), c: cv("--qc") };
      const soft = mode === "text" ? cv("--qb-soft") : cv("--qc-soft");
      const mx = (cur.a[0] + cur[near][0]) / 2, my = (cur.a[1] + cur[near][1]) / 2;
      s("ellipse", { cx: mx, cy: my, rx: 62, ry: 40, fill: soft, opacity: 0.85 }, svg);
      s("line", { x1: cur.a[0], y1: cur.a[1], x2: cur[far][0], y2: cur[far][1], stroke: col[far], "stroke-width": 1.5, "stroke-dasharray": "5 4", opacity: 0.7 }, svg);
      s("line", { x1: cur.a[0], y1: cur.a[1], x2: cur[near][0], y2: cur[near][1], stroke: col[near], "stroke-width": 3 }, svg);
      for (const k of ["a", "b", "c"]) {
        s("circle", { cx: cur[k][0], cy: cur[k][1], r: 11, fill: col[k] }, svg);
        s("text", { x: cur[k][0], y: cur[k][1] + 4.5, "text-anchor": "middle", "font-size": 12, "font-weight": 700, fill: "#fff", text: k }, svg).style.fill = "#fff";
      }
      s("text", { x: 12, y: 20, "font-size": 12, text: mode === "text" ? "close = queries look alike" : "close = answers overlap" }, svg);
    }
    function go(m) {
      mode = m;
      const from = JSON.parse(JSON.stringify(cur)), to = layouts[m];
      const t0 = performance.now();
      cancelAnimationFrame(anim);
      const step = (t) => {
        const u = Math.min(1, (t - t0) / 650), e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
        for (const k in to) cur[k] = [from[k][0] + (to[k][0] - from[k][0]) * e, from[k][1] + (to[k][1] - from[k][1]) * e];
        draw();
        if (u < 1) anim = requestAnimationFrame(step);
      };
      anim = requestAnimationFrame(step);
      const msg = $("#teaserMsg");
      if (m === "text") {
        msg.className = "verdict bad";
        msg.innerHTML = `Top match for (a): <b style="color:var(--qb)">(b)</b>, which shares only 37 answers (<i>J</i> = 0.19). A <b>poor</b> candidate for reuse.`;
      } else {
        msg.className = "verdict good";
        msg.innerHTML = `Top match for (a): <b style="color:var(--qc)">(c)</b>, which returns the same 136 answers (<i>J</i> = 1.00). A <b>safe</b> candidate for reuse.`;
      }
    }
    segment("teaserSeg", (d) => go(d.mode));
    onTheme(draw);
    go("text");
  })();

  /* ------------------------------------------------------------------ 1b. scatter */
  (function scatter() {
    if (!$("#scatterSvg")) return;
    const svg = $("#scatterSvg");
    let g = "dbpedia";
    const W = 480, H = 330, m = { l: 46, r: 12, t: 12, b: 40 };
    const X = (x) => m.l + x * (W - m.l - m.r), Y = (y) => H - m.b - y * (H - m.t - m.b);
    function draw() {
      clear(svg);
      const data = D[g].scatter;
      s("rect", { x: X(0), y: Y(1), width: X(0.5) - X(0), height: Y(0.8) - Y(1), fill: cv("--good-soft") }, svg);
      s("rect", { x: X(0.5), y: Y(0.1), width: X(1) - X(0.5), height: Y(0) - Y(0.1), fill: cv("--bad-soft") }, svg);
      for (let i = 0; i <= 5; i++) {
        const v = i / 5;
        s("line", { x1: X(0), x2: X(1), y1: Y(v), y2: Y(v), stroke: cv("--line"), "stroke-width": 0.6 }, svg);
        s("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end", "font-size": 10, text: v.toFixed(1) }, svg);
        s("text", { x: X(v), y: H - m.b + 15, "text-anchor": "middle", "font-size": 10, text: v.toFixed(1) }, svg);
      }
      s("text", { x: (X(0) + X(1)) / 2, y: H - 6, "text-anchor": "middle", "font-size": 11, text: "text similarity (TF-IDF cosine)" }, svg);
      s("text", { x: 12, y: (Y(0) + Y(1)) / 2, "text-anchor": "middle", "font-size": 11, transform: `rotate(-90 12 ${(Y(0) + Y(1)) / 2})`, text: "answer overlap J" }, svg);
      const cg = cv("--good"), cb = cv("--bad"), cm = cv("--mid");
      const r = rng(3);
      for (const [x, y] of data) {
        const jx = (r() - 0.5) * 0.006, jy = (r() - 0.5) * 0.012;
        s("circle", { cx: X(Math.min(1, Math.max(0, x + jx))), cy: Y(Math.min(1, Math.max(0, y + jy))), r: 2.3, fill: y >= 0.8 ? cg : y <= 0.1 ? cb : cm, opacity: 0.5 }, svg);
      }
      s("text", { x: X(0.02), y: Y(0.97), "font-size": 10.5, "font-weight": 600, text: "same answers, different text" }, svg).style.fill = cg;
      s("text", { x: X(0.98), y: Y(0.03) - 4, "text-anchor": "end", "font-size": 10.5, "font-weight": 600, text: "similar text, different answers" }, svg).style.fill = cb;
      const hi = data.filter((p) => p[0] >= 0.5), pos = data.filter((p) => p[1] >= 0.8);
      $("#scA").textContent = hi.length ? pct(hi.filter((p) => p[1] <= 0.1).length / hi.length) : "–";
      $("#scB").textContent = pos.length ? pct(pos.filter((p) => p[0] < 0.5).length / pos.length) : "–";
      $("#scNote").innerHTML = `Random sample of ${data.length.toLocaleString()} of the ${D[g].stored_pairs.toLocaleString()} stored ${g === "dbpedia" ? "DBpedia" : "Wikidata"} test pairs. Pooled Spearman ρ of TF-IDF with <i>J</i> over all pairs is ${fmt(D[g].tfidf_spearman)}, similar to zero-shot encoders (${fmt(P.alignment[g].zs)}): a good overall correlation still hides many wrong top matches.`;
    }
    segment("scatterSeg", (d) => { g = d.g; draw(); });
    onTheme(draw);
  })();

  /* ------------------------------------------------------------------ 2. steps */
  (function steps() {
    if (!$("#steps")) return;
    const items = $$("#steps li");
    items.forEach((li) => li.addEventListener("click", () => {
      items.forEach((x) => x.classList.toggle("on", x === li));
      $$(".step-panel").forEach((p) => (p.hidden = p.dataset.panel !== li.dataset.step));
      if (li.dataset.step === "3") window.dispatchEvent(new Event("toyshow"));
    }));
  })();

  /* ------------------------------------------------------------------ 2b. Jaccard / Venn */
  (function venn() {
    if (!$("#vennSvg")) return;
    const svg = $("#vennSvg");
    const a = $("#vA"), b = $("#vB"), i = $("#vI");
    function lensArea(r1, r2, d) {
      if (d >= r1 + r2) return 0;
      if (d <= Math.abs(r1 - r2)) return Math.PI * Math.min(r1, r2) ** 2;
      const a1 = r1 * r1 * Math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1));
      const a2 = r2 * r2 * Math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2));
      return a1 + a2 - 0.5 * Math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2));
    }
    function update(ev) {
      let A = +a.value, B = +b.value, I = +i.value;
      I = Math.min(I, A, B);
      i.value = I;
      $("#oA").textContent = A; $("#oB").textContent = B; $("#oI").textContent = I;
      const U = A + B - I, J = U ? I / U : 0;
      const k = 70 / Math.sqrt(200 / Math.PI);
      const r1 = Math.sqrt(A / Math.PI) * k, r2 = Math.sqrt(B / Math.PI) * k;
      const want = I * k * k; // circle area is proportional to the number of rows
      let lo = Math.abs(r1 - r2), hi = r1 + r2;
      for (let t = 0; t < 40; t++) { const mid = (lo + hi) / 2; if (lensArea(r1, r2, mid) > want) lo = mid; else hi = mid; }
      const d = I === 0 ? r1 + r2 + 8 : (lo + hi) / 2;
      clear(svg);
      const cx = 180 - d / 2 + (r1 - r2) / 4;
      s("circle", { cx: cx, cy: 100, r: r1, fill: cv("--qa"), opacity: 0.28, stroke: cv("--qa"), "stroke-width": 2 }, svg);
      s("circle", { cx: cx + d, cy: 100, r: r2, fill: cv("--qb"), opacity: 0.28, stroke: cv("--qb"), "stroke-width": 2 }, svg);
      s("text", { x: cx - r1 * 0.45, y: 104, "text-anchor": "middle", "font-size": 12, "font-weight": 600, text: A - I }, svg);
      s("text", { x: cx + d + r2 * 0.45, y: 104, "text-anchor": "middle", "font-size": 12, "font-weight": 600, text: B - I }, svg);
      if (I > 0) s("text", { x: cx + (d + r1 - r2) / 2, y: 104, "text-anchor": "middle", "font-size": 12, "font-weight": 700, text: I }, svg);
      s("text", { x: 10, y: 190, "font-size": 11, text: "Ans(qᵢ)" }, svg).style.fill = cv("--qa");
      s("text", { x: 350, y: 190, "text-anchor": "end", "font-size": 11, text: "Ans(qⱼ)" }, svg).style.fill = cv("--qb");
      $("#jVal").textContent = J.toFixed(2);
      const lab = $("#jLabel");
      if (J >= 0.8) { lab.className = "pill pos"; lab.textContent = "positive pair: pull together"; }
      else if (J <= 0.1) { lab.className = "pill neg"; lab.textContent = "negative pair: push apart"; }
      else { lab.className = "pill mid"; lab.textContent = "unlabeled: not used in training"; }
      $("#jMarker").style.left = `calc(${J * 100}% - 1.5px)`;
      const prec = B ? I / B : 0, rec = A ? I / A : 0;
      $("#jGuarantee").innerHTML = `If q<sub>j</sub>'s cached answers were reused for q<sub>i</sub>: <b>${pct(prec)}</b> of returned rows are correct and <b>${pct(rec)}</b> of the needed rows are returned. Both are always ≥ <i>J</i>, so a positive (<i>J</i> ≥ 0.8) guarantees at least 80% of each.`;
    }
    [a, b, i].forEach((x) => x.addEventListener("input", update));
    $$(".presets button").forEach((btn) => btn.addEventListener("click", () => {
      const [A, B, I] = btn.dataset.p.split(",").map(Number);
      a.value = A; b.value = B; i.value = I; update();
    }));
    onTheme(update);
  })();

  /* ------------------------------------------------------------------ 2c. split */
  (function split() {
    if (!$("#splitSvg")) return;
    const svg = $("#splitSvg");
    const comps = [5, 3, 4, 2, 3, 2, 4, 2, 3, 2];
    const nodes = [], edges = [];
    let x0 = 30;
    comps.forEach((size, ci) => {
      const cx = x0 + 22, cy = 115, R = size <= 2 ? 16 : 26;
      const ids = [];
      for (let k = 0; k < size; k++) {
        const ang = (k / size) * Math.PI * 2 - Math.PI / 2 + ci;
        nodes.push({ x: cx + R * Math.cos(ang), y: cy + R * Math.sin(ang) * 1.6, c: ci });
        ids.push(nodes.length - 1);
      }
      for (let k = 1; k < ids.length; k++) edges.push([ids[k - 1], ids[k]]);
      if (ids.length >= 4) edges.push([ids[0], ids[ids.length - 1]]);
      x0 += 2 * R + 30;
    });
    const scale = 600 / x0;
    nodes.forEach((n) => { n.x = 20 + (n.x - 20) * scale; });
    const compSplit = ["tr", "tr", "te", "tr", "tr", "va", "tr", "tr", "tr", "tr"];
    let mode = "component";
    function draw() {
      clear(svg);
      let assign;
      if (mode === "component") assign = nodes.map((n) => compSplit[n.c]);
      else {
        const r = rng(11);
        assign = nodes.map(() => { const u = r(); return u < 0.6 ? "tr" : u < 0.8 ? "va" : "te"; });
      }
      const col = { tr: cv("--tr"), va: cv("--va"), te: cv("--te") };
      let leaks = 0;
      for (const [p, q] of edges) {
        const leak = assign[p] !== assign[q];
        if (leak) leaks++;
        s("line", { x1: nodes[p].x, y1: nodes[p].y, x2: nodes[q].x, y2: nodes[q].y, stroke: leak ? cv("--bad") : cv("--ink-3"), "stroke-width": leak ? 3 : 1.6, "stroke-dasharray": leak ? "5 3" : "" }, svg);
      }
      nodes.forEach((n, k) => s("circle", { cx: n.x, cy: n.y, r: 8, fill: col[assign[k]], stroke: cv("--surface"), "stroke-width": 2 }, svg));
      s("text", { x: 10, y: 18, "font-size": 10, text: "nodes = queries · edges = positive pairs (J ≥ 0.8) · each cluster = one connected component" }, svg);
      const msg = $("#splitMsg");
      if (leaks) { msg.className = "verdict bad"; msg.innerHTML = `<b>${leaks} of ${edges.length}</b> positive pairs cross splits: the model would be tested on relations it saw in training.`; }
      else { msg.className = "verdict good"; msg.innerHTML = `<b>0 of ${edges.length}</b> positive pairs cross splits. SQuARE has 1,156 (DBpedia) and 3,846 (Wikidata) such components, split 80/10/10.`; }
    }
    segment("splitSeg", (d) => { mode = d.s; draw(); });
    onTheme(draw);
  })();

  /* ------------------------------------------------------------------ 2d. toy contrastive training */
  (function toy() {
    if (!$("#toySvg")) return;
    const svg = $("#toySvg"), lossSvg = $("#lossSvg");
    const Q = [
      // group 0: actors born in Paris (same answers)
      ["?p a Actor ; birthPlace Paris", 0, [-0.35, 0.10]],
      ["?p birthPlace ?c FILTER(?c=Paris)", 0, [0.30, -0.45]],
      ["?p birthPlace Paris . ?p a Actor", 0, [-0.25, 0.25]],
      // group 1: actors who died in Paris
      ["?p a Actor ; deathPlace Paris", 1, [-0.20, 0.05]],
      ["?p deathPlace ?c FILTER(?c=Paris)", 1, [0.40, -0.35]],
      ["?p deathPlace Paris . ?p a Actor", 1, [-0.10, 0.22]],
      // group 2: actors born in Berlin
      ["?p a Actor ; birthPlace Berlin", 2, [-0.40, -0.05]],
      ["?p birthPlace ?c FILTER(?c=Berlin)", 2, [0.25, -0.55]],
      ["?p birthPlace Berlin . ?p a Actor", 2, [-0.30, 0.35]],
      // group 3: films directed by Varda
      ["?f director Agnès_Varda", 3, [0.55, 0.40]],
      ["?f a Film ; director Agnès_Varda", 3, [0.10, 0.50]],
      ["?f director ?d FILTER(?d=Varda)", 3, [0.45, -0.20]],
    ];
    const groupCol = () => [cv("--qa"), cv("--qb"), cv("--qc"), cv("--te")];
    let Z, epoch, losses, hardPool, mined, playing = false, raf = null, stepInEpoch = 0, R;
    const STEPS = 14, TAU = 0.35, LR = 0.035, M = 3;
    function reset() {
      Z = Q.map((q) => q[2].slice());
      epoch = 0; losses = []; hardPool = null; mined = []; stepInEpoch = 0; R = rng(5);
      playing = false; $("#toyPlay").textContent = "▶ Train";
      cancelAnimationFrame(raf);
      draw();
    }
    const sim = (i, j) => -((Z[i][0] - Z[j][0]) ** 2 + (Z[i][1] - Z[j][1]) ** 2);
    function mine() {
      hardPool = Q.map((q, i) => Q.map((_, j) => j).filter((j) => Q[j][1] !== q[1]).sort((x, y) => sim(i, y) - sim(i, x)).slice(0, 3));
      mined = [];
      hardPool.forEach((pool, i) => mined.push([i, pool[0]]));
    }
    function trainStep() {
      const n = Q.length, G = Z.map(() => [0, 0]);
      let L = 0;
      const pos = [], negs = [];
      for (let i = 0; i < n; i++) {
        const ps = Q.map((_, j) => j).filter((j) => j !== i && Q[j][1] === Q[i][1]);
        pos.push(ps[Math.floor(R() * ps.length)]);
        const all = Q.map((_, j) => j).filter((j) => Q[j][1] !== Q[i][1]);
        let chosen = [];
        if (hardPool && $("#toyHard").checked) {
          const h = hardPool[i].slice().sort(() => R() - 0.5).slice(0, 2);
          chosen = h;
        }
        const rest = all.filter((j) => !chosen.includes(j)).sort(() => R() - 0.5);
        while (chosen.length < M) chosen.push(rest.pop());
        negs.push(chosen);
      }
      for (let i = 0; i < n; i++) {
        // candidates: all positives in the batch + all explicit negatives; mask known positives of i
        const cand = new Set([...pos, ...negs.flat()]);
        cand.delete(i);
        const list = [...cand].filter((c) => c === pos[i] || Q[c][1] !== Q[i][1]);
        const logits = list.map((c) => sim(i, c) / TAU);
        const mx = Math.max(...logits);
        const ex = logits.map((v) => Math.exp(v - mx)), Zs = ex.reduce((a, b) => a + b, 0);
        const w = ex.map((v) => v / Zs);
        const pi = list.indexOf(pos[i]);
        L += -Math.log(w[pi] + 1e-12);
        list.forEach((c, k) => {
          const coef = (w[k] - (k === pi ? 1 : 0)) / TAU; // dL/dlogit * dlogit/dsim
          const dx = Z[i][0] - Z[c][0], dy = Z[i][1] - Z[c][1];
          // sim = -(dx^2+dy^2): dsim/dZi = -2d, dsim/dZc = +2d
          G[i][0] += coef * -2 * dx; G[i][1] += coef * -2 * dy;
          G[c][0] += coef * 2 * dx; G[c][1] += coef * 2 * dy;
        });
      }
      for (let i = 0; i < n; i++) {
        Z[i][0] -= LR * (G[i][0] / n + 0.02 * Z[i][0]);
        Z[i][1] -= LR * (G[i][1] / n + 0.02 * Z[i][1]);
      }
      return L / n;
    }
    function draw() {
      clear(svg);
      const W = 420, H = 300;
      let xs = Z.map((z) => z[0]), ys = Z.map((z) => z[1]);
      const cx = (Math.max(...xs) + Math.min(...xs)) / 2, cy = (Math.max(...ys) + Math.min(...ys)) / 2;
      const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 0.6);
      const k = 230 / span;
      const px = (z) => [W / 2 + (z[0] - cx) * k, H / 2 + (z[1] - cy) * k * 0.95];
      s("rect", { x: 0, y: 0, width: W, height: H, rx: 10, fill: cv("--surface-2") }, svg);
      const col = groupCol();
      if (epoch > 0 && mined.length && $("#toyHard").checked) {
        for (const [i, j] of mined) {
          const a = px(Z[i]), b = px(Z[j]);
          s("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: cv("--bad"), "stroke-width": 1.2, "stroke-dasharray": "3 3", opacity: 0.75 }, svg);
        }
      }
      Z.forEach((z, i) => {
        const [x, y] = px(z);
        const c = s("circle", { cx: x, cy: y, r: 7.5, fill: col[Q[i][1]], stroke: cv("--surface"), "stroke-width": 1.5 }, svg);
        const t = s("title", {}, c); t.textContent = Q[i][0];
      });
      if (epoch === 0) {
        Z.forEach((z, i) => {
          if (i % 3 !== 0) return;
          const [x, y] = px(z);
          s("text", { x: x + 10, y: y + 4, "font-size": 9.5, text: Q[i][0] }, svg);
        });
      }
      if (epoch >= 6) {
        const names = ["actors born in Paris", "actors died in Paris", "actors born in Berlin", "films by Varda"];
        for (let gi = 0; gi < 4; gi++) {
          const pts = Z.filter((_, i) => Q[i][1] === gi).map(px);
          const mx = pts.reduce((a, p) => a + p[0], 0) / pts.length, my = pts.reduce((a, p) => a + p[1], 0) / pts.length;
          const left = mx > W / 2;
          s("text", { x: mx + (left ? -14 : 14), y: my + 4, "text-anchor": left ? "end" : "start", "font-size": 11, "font-weight": 600, text: names[gi] }, svg).style.fill = col[gi];
        }
      }
      s("text", { x: 10, y: H - 10, "font-size": 10.5, text: epoch === 0 ? "start: text-based layout (hover a dot for its query)" : "colour = same answer set" }, svg);
      $("#toyEpoch").textContent = epoch;
      // loss
      clear(lossSvg);
      const lw = 480, lh = 150, ml = 34, mb = 20;
      s("text", { x: 0, y: 12, "font-size": 12, text: "training loss (InfoNCE)" }, lossSvg);
      s("line", { x1: ml, x2: lw, y1: lh - mb, y2: lh - mb, stroke: cv("--line") }, lossSvg);
      if (losses.length > 1) {
        const mxL = Math.max(...losses), mnL = 0;
        const pts = losses.map((v, j) => `${ml + (j / 20) * (lw - ml)},${lh - mb - ((v - mnL) / (mxL - mnL || 1)) * (lh - mb - 16)}`).join(" ");
        s("polyline", { points: pts, fill: "none", stroke: cv("--accent"), "stroke-width": 2 }, lossSvg);
        s("text", { x: ml - 4, y: 26, "text-anchor": "end", "font-size": 11, text: mxL.toFixed(1) }, lossSvg);
      }
      s("text", { x: lw, y: lh - 4, "text-anchor": "end", "font-size": 11, text: "epoch 20" }, lossSvg);
    }
    function frame() {
      if (!playing) return;
      if (stepInEpoch === 0) {
        epoch++;
        if (epoch >= 2 && (epoch - 2) % 2 === 0) mine();
      }
      const L = trainStep();
      stepInEpoch++;
      if (stepInEpoch >= STEPS) { losses.push(L); stepInEpoch = 0; }
      draw();
      if (epoch >= 20 && stepInEpoch === 0) { playing = false; $("#toyPlay").textContent = "✓ Done"; return; }
      raf = requestAnimationFrame(frame);
    }
    $("#toyPlay").addEventListener("click", () => {
      if (epoch >= 20) reset();
      playing = !playing;
      $("#toyPlay").textContent = playing ? "❚❚ Pause" : "▶ Train";
      if (playing) raf = requestAnimationFrame(frame);
    });
    $("#toyReset").addEventListener("click", reset);
    $("#toyHard").addEventListener("change", draw);
    reset();
    redrawers.push(draw);
  })();

  /* ------------------------------------------------------------------ 3. explorer */
  (function explorer() {
    if (!$("#exList")) return;
    let g = "dbpedia", idx = 0;
    const jBadge = (J) => J == null ? `<span class="jb mid">partial</span>` : J >= 0.8 ? `<span class="jb pos">${fmt(J)} ✓</span>` : J <= 0.1 ? `<span class="jb neg">${fmt(J)} ✗</span>` : `<span class="jb mid">${fmt(J)}</span>`;
    function list() {
      const ex = D[g].examples;
      $("#exList").innerHTML = ex.map((e, k) => {
        const r = e.text_rank_of_first_match;
        const first = e.anchor.split("\n").slice(1).join(" ").replace(/\s+/g, " ").replace(/^WHERE\s*/, "").slice(0, 70);
        return `<button class="exitem${k === idx ? " on" : ""}" data-k="${k}"><span class="r ${r === 1 ? "good" : "bad"}">${r === 1 ? "rank 1" : "rank " + r}</span>${esc(first)}</button>`;
      }).join("");
    }
    function show() {
      const e = D[g].examples[idx];
      $("#exAnchor").textContent = e.anchor;
      $("#exAnchorAns").textContent = `${(e.anchor_answers || 0).toLocaleString()} answer row${e.anchor_answers === 1 ? "" : "s"} · ${e.n_relevant} high-overlap quer${e.n_relevant === 1 ? "y" : "ies"} among the ${(D[g].test_queries - 1).toLocaleString()} other test queries`;
      const r = e.text_rank_of_first_match;
      const v = $("#exVerdict");
      if (r === 1) { v.className = "verdict good"; v.innerHTML = `Here text similarity works: its top match has high answer overlap. Such cases are common on DBpedia, where 98% of test queries share a template with training.`; }
      else { v.className = "verdict bad"; v.innerHTML = `Text similarity ranks the first high-overlap query at <b>rank ${r}</b>. Its top match${r > 2 ? "es look" : " looks"} closer but return${r > 2 ? "" : "s"} different answers. Results-aware training targets exactly this ordering.`; }
      const rows = [];
      let last = 0;
      for (const c of e.candidates.slice().sort((x, y) => x.rank - y.rank)) {
        if (c.rank > last + 1) rows.push(`<tr class="gap"><td colspan="5">⋯ ${c.rank - last - 1} more ⋯</td></tr>`);
        last = c.rank;
        rows.push(`<tr class="${c.best ? "best" : ""}"><td>${c.rank}${c.best ? ' <span class="star">★</span>' : ""}</td><td class="q">${esc(c.q.replace(/\s+/g, " "))}</td><td>${(c.answers || 0).toLocaleString()}</td><td>${fmt(c.tfidf)}</td><td>${jBadge(c.J)}</td></tr>`);
      }
      $("#exRows").innerHTML = rows.join("");
    }
    $("#exList").addEventListener("click", (ev) => {
      const b = ev.target.closest(".exitem"); if (!b) return;
      idx = +b.dataset.k; list(); show();
    });
    segment("exSeg", (d) => { g = d.g; idx = 0; list(); show(); });
    list(); show();
  })();

  /* ------------------------------------------------------------------ charts helpers */
  function hbars(svg, rows, opt) {
    // rows: [label, value, kind]
    clear(svg);
    const W = 480, rowH = opt.rowH || 34, ml = opt.ml || 190, mr = 46, top = 8;
    const H = top + rows.length * rowH + 22;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const X = (v) => ml + v * (W - ml - mr);
    for (let i = 0; i <= 4; i++) {
      const v = i / 4 * (opt.max || 1);
      s("line", { x1: X(v / (opt.max || 1)), x2: X(v / (opt.max || 1)), y1: top, y2: H - 20, stroke: cv("--line"), "stroke-width": 0.6 }, svg);
      s("text", { x: X(v / (opt.max || 1)), y: H - 6, "text-anchor": "middle", "font-size": 10, text: v.toFixed(2) }, svg);
    }
    const kindCol = { base: cv("--mid"), zs: cv("--qa"), ft: cv("--qc"), syn: cv("--mid"), ans: cv("--qc"), tf: cv("--mid") };
    rows.forEach(([lab, v, kind], i) => {
      const y = top + i * rowH;
      s("text", { x: ml - 8, y: y + rowH / 2 + 4, "text-anchor": "end", "font-size": 11.5, "font-weight": kind === "ft" || kind === "ans" ? 700 : 400, text: lab }, svg);
      const bw = Math.max(1, X(v / (opt.max || 1)) - ml);
      s("rect", { x: ml, y: y + 6, width: bw, height: rowH - 12, rx: 4, fill: kindCol[kind] || cv("--mid"), opacity: kind === "base" || kind === "tf" || kind === "syn" ? 0.55 : 1 }, svg);
      s("text", { x: ml + bw + 6, y: y + rowH / 2 + 4, "font-size": 11.5, "font-weight": 600, text: fmt(v) }, svg);
    });
  }

  /* ------------------------------------------------------------------ 3b. complete-label bars */
  (function fullCollection() {
    if (!$("#fcSvg")) return;
    let g = "dbpedia";
    const draw = () => hbars($("#fcSvg"), P.retrievalComplete[g].map((r) => [r[0].replace(" (mean of 10)", ""), r[1], r[4]]), { ml: 200, rowH: 36 });
    segment("fcSeg", (d) => { g = d.g; draw(); });
    onTheme(draw);
  })();

  /* ------------------------------------------------------------------ 3c. dumbbell */
  (function dumbbell() {
    if (!$("#dumbSvg")) return;
    const svg = $("#dumbSvg");
    let g = "dbpedia";
    function draw() {
      clear(svg);
      const R = P.retrievalStored[g];
      const W = 480, rowH = 29, ml = 130, mr = 20, top = 22;
      const H = top + P.encoders.length * rowH + 26;
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      const max = 0.7;
      const X = (v) => ml + (v / max) * (W - ml - mr);
      for (let v = 0; v <= max + 1e-9; v += 0.1) {
        s("line", { x1: X(v), x2: X(v), y1: top - 6, y2: H - 22, stroke: cv("--line"), "stroke-width": 0.6 }, svg);
        s("text", { x: X(v), y: H - 8, "text-anchor": "middle", "font-size": 10, text: v.toFixed(1) }, svg);
      }
      const tf = R.baselines["TF-IDF"];
      s("line", { x1: X(tf), x2: X(tf), y1: top - 10, y2: H - 22, stroke: cv("--ink-3"), "stroke-dasharray": "4 3", "stroke-width": 1.2 }, svg);
      s("text", { x: X(tf), y: 10, "text-anchor": "middle", "font-size": 10, text: "TF-IDF " + fmt(tf) }, svg);
      P.encoders.forEach((name, i) => {
        const y = top + i * rowH + rowH / 2;
        const zs = R.zs[i], ft = zs + R.gain[i];
        s("text", { x: ml - 8, y: y + 4, "text-anchor": "end", "font-size": 11.5, text: name }, svg);
        s("line", { x1: X(zs), x2: X(ft), y1: y, y2: y, stroke: cv("--qc"), "stroke-width": 3, opacity: 0.45 }, svg);
        s("circle", { cx: X(zs), cy: y, r: 5.5, fill: cv("--surface"), stroke: cv("--qa"), "stroke-width": 2 }, svg);
        s("circle", { cx: X(ft), cy: y, r: 6, fill: cv("--qc") }, svg);
        s("text", { x: X(ft) + 9, y: y + 4, "font-size": 10.5, "font-weight": 600, text: "+" + fmt(R.gain[i]) }, svg).style.fill = cv("--good");
      });
      const ly = 8, lx = 14;
      s("circle", { cx: lx, cy: ly, r: 5, fill: cv("--surface"), stroke: cv("--qa"), "stroke-width": 2 }, svg);
      s("text", { x: lx + 9, y: ly + 4, "font-size": 10.5, text: "zero-shot" }, svg);
      s("circle", { cx: lx + 80, cy: ly, r: 5, fill: cv("--qc") }, svg);
      s("text", { x: lx + 89, y: ly + 4, "font-size": 10.5, text: "fine-tuned" }, svg);
    }
    segment("dbSeg", (d) => { g = d.g; draw(); });
    onTheme(draw);
  })();

  /* ------------------------------------------------------------------ 3d. syntax control + seen/unseen */
  function pairedBars(svg, groups) {
    // groups: [{label, bars:[[name, value, kind]]}]
    clear(svg);
    const W = 480, H = 190, ml = 10, top = 14, bottom = 40;
    const gw = (W - ml) / groups.length;
    const Y = (v) => H - bottom - v * (H - top - bottom) / 0.8;
    for (let v = 0; v <= 0.8 + 1e-9; v += 0.2) {
      s("line", { x1: ml, x2: W, y1: Y(v), y2: Y(v), stroke: cv("--line"), "stroke-width": 0.6 }, svg);
    }
    const col = { syn: cv("--mid"), ans: cv("--qc"), tf: cv("--mid"), ft: cv("--qc") };
    groups.forEach((gr, gi) => {
      const x0 = ml + gi * gw;
      const bw = Math.min(70, (gw - 40) / gr.bars.length);
      gr.bars.forEach(([name, v, kind], bi) => {
        const x = x0 + (gw - bw * gr.bars.length - 10) / 2 + bi * (bw + 10);
        s("rect", { x, y: Y(v), width: bw, height: Y(0) - Y(v), rx: 4, fill: col[kind], opacity: kind === "syn" || kind === "tf" ? 0.55 : 1 }, svg);
        s("text", { x: x + bw / 2, y: Y(v) - 5, "text-anchor": "middle", "font-size": 11.5, "font-weight": 700, text: fmt(v) }, svg);
        s("text", { x: x + bw / 2, y: Y(0) + 13, "text-anchor": "middle", "font-size": 10, text: name }, svg);
      });
      s("text", { x: x0 + gw / 2, y: H - 6, "text-anchor": "middle", "font-size": 11.5, "font-weight": 600, text: gr.label }, svg);
    });
  }
  if ($("#ctrlSvg")) onTheme(() => {
    const sc = P.syntaxControl;
    pairedBars($("#ctrlSvg"), [
      { label: "DBpedia  (+" + sc.dbpedia.diff.split(" ")[0] + ")", bars: [["syntax labels", sc.dbpedia.syntax, "syn"], ["answer labels", sc.dbpedia.answers, "ans"]] },
      { label: "Wikidata  (+" + sc.wikidata.diff.split(" ")[0] + ")", bars: [["syntax labels", sc.wikidata.syntax, "syn"], ["answer labels", sc.wikidata.answers, "ans"]] },
    ]);
    const su = P.seenUnseen;
    pairedBars($("#suSvg"), [
      { label: `seen templates (n = ${su.seen.n})`, bars: [["TF-IDF", su.seen.tfidf, "tf"], ["fine-tuned", su.seen.ft, "ft"]] },
      { label: `unseen templates (n = ${su.unseen.n})`, bars: [["TF-IDF", su.unseen.tfidf, "tf"], ["fine-tuned", su.unseen.ft, "ft"]] },
    ]);
  });

  /* ------------------------------------------------------------------ 4. cache walk-through */
  (function cacheSim() {
    if (!$("#csStage")) return;
    const Qs = {
      a: { k: "qa", t: "?p a dbo:Actor ; dbo:birthPlace dbr:Paris", n: 136 },
      b: { k: "qb", t: "?p a dbo:Actor ; dbo:deathPlace dbr:Paris", n: 94 },
      c: { k: "qc", t: "?p a dbo:Actor ; dbo:birthPlace ?c .\nFILTER(?c = dbr:Paris)", n: 136 },
      d: { k: "qd", t: "?p a dbo:Actor ; dbo:birthPlace dbr:Berlin", n: null },
    };
    // stream: [incoming, nearest cached, cos(text), cos(ours), J(incoming, nearest)]
    const stream = [
      ["a", null, 0, 0, null],
      ["b", "a", 0.93, 0.41, 0.19],
      ["c", "a", 0.74, 0.94, 1.0],
      ["d", "a", 0.95, 0.38, 0.0],
    ];
    const theta = 0.8;
    let mode = "text", pos = 0, cache = [], tally;
    const card = (id, extra = "") => `<div class="cs-q ${Qs[id].k} ${extra}"><b>(${id})</b> ${esc(Qs[id].t)}</div>`;
    function reset() { pos = 0; cache = []; tally = { hits: 0, ok: 0, bad: 0, exec: 0 }; render(null); $("#csNext").disabled = false; $("#csNext").textContent = "Next query ▶"; }
    function render(ev) {
      let inc = "<h4>incoming query</h4>", dec = "<h4>decision (θ = 0.8)</h4>";
      if (!ev) { inc += `<p class="note">Press “Next query”.</p>`; dec += ""; }
      else {
        const [id, near, ct, co, J] = ev;
        inc += card(id, "new");
        if (!near) {
          dec += `<div class="cs-dec"><div class="big ex">cache empty → execute</div>store (${id}) and its answers</div>`;
        } else {
          const c = mode === "text" ? ct : co;
          dec += `<div class="cs-dec">nearest cached: <b>(${near})</b>, cosine <b>${fmt(c)}</b>`;
          if (c >= theta) {
            const ok = J >= 0.8;
            dec += `<div class="big ${ok ? "ok" : "ko"}">${ok ? "✓ reuse, correct" : "✗ reuse, wrong answers"}</div>J((${id}),(${near})) = ${fmt(J)}${J === 0 ? " (different people)" : ""}`;
          } else {
            dec += `<div class="big ex">below θ → execute</div>${J >= 0.8 ? `missed a safe reuse (J = ${fmt(J)})` : `correctly avoided a bad reuse (J = ${fmt(J)})`}`;
          }
          dec += `</div>`;
        }
      }
      dec += `<div class="tally"><span>hits ${tally.hits}</span><span style="color:var(--good)">correct ${tally.ok}</span><span style="color:var(--bad)">wrong ${tally.bad}</span><span>executed ${tally.exec}</span></div>`;
      const cch = `<h4>cache (FIFO, K = 128)</h4>` + (cache.length ? cache.map((id) => card(id)).join("") : `<p class="note">empty</p>`);
      $("#csStage").innerHTML = `<div class="cs-col">${inc}</div><div class="cs-col">${dec}</div><div class="cs-col">${cch}</div>`;
    }
    function next() {
      if (pos >= stream.length) return;
      const ev = stream[pos++];
      const [id, near, ct, co, J] = ev;
      const c = mode === "text" ? ct : co;
      if (near && c >= theta) { tally.hits++; if (J >= 0.8) tally.ok++; else tally.bad++; }
      else { tally.exec++; cache.push(id); }
      render(ev);
      if (pos >= stream.length) { $("#csNext").disabled = true; $("#csNext").textContent = "End of stream"; }
    }
    $("#csNext").addEventListener("click", next);
    $("#csReset").addEventListener("click", reset);
    segment("csSeg", (d) => { mode = d.m; reset(); });
    reset();
  })();

  /* ------------------------------------------------------------------ 4b. cache accuracy bars */
  (function cacheAcc() {
    if (!$("#caSvg")) return;
    const svg = $("#caSvg");
    let g = "dbpedia";
    function draw() {
      clear(svg);
      const C = P.caching[g];
      const W = 480, rowH = 29, ml = 130, mr = 40, top = 22;
      const H = top + P.encoders.length * rowH + 26;
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      const X = (v) => ml + v * (W - ml - mr);
      for (let v = 0; v <= 1.0001; v += 0.2) {
        s("line", { x1: X(v), x2: X(v), y1: top - 6, y2: H - 22, stroke: cv("--line"), "stroke-width": 0.6 }, svg);
        s("text", { x: X(v), y: H - 8, "text-anchor": "middle", "font-size": 10, text: v.toFixed(1) }, svg);
      }
      P.encoders.forEach((name, i) => {
        const y = top + i * rowH;
        s("text", { x: ml - 8, y: y + rowH / 2 + 4, "text-anchor": "end", "font-size": 11.5, text: name }, svg);
        s("rect", { x: ml, y: y + 5, width: Math.max(1, X(C.ftAcc[i]) - ml), height: rowH - 10, rx: 4, fill: cv("--qc") }, svg);
        s("rect", { x: ml, y: y + 9, width: Math.max(1, X(C.zsAcc[i]) - ml), height: rowH - 18, rx: 3, fill: cv("--qa"), opacity: 0.85 }, svg);
        s("text", { x: X(C.ftAcc[i]) + 6, y: y + rowH / 2 + 4, "font-size": 10.5, "font-weight": 600, text: fmt(C.ftAcc[i]) }, svg);
      });
      s("rect", { x: W - 190, y: 2, width: 12, height: 10, rx: 2, fill: cv("--qa") }, svg);
      s("text", { x: W - 174, y: 11, "font-size": 10.5, text: "zero-shot" }, svg);
      s("rect", { x: W - 110, y: 2, width: 12, height: 10, rx: 2, fill: cv("--qc") }, svg);
      s("text", { x: W - 94, y: 11, "font-size": 10.5, text: "fine-tuned" }, svg);
    }
    segment("caSeg", (d) => { g = d.g; draw(); });
    onTheme(draw);
  })();

  /* ------------------------------------------------------------------ 4c. risk-coverage with theta slider */
  (function riskCov() {
    if (!$("#rcSvg")) return;
    const svg = $("#rcSvg"), slider = $("#thr");
    let g = "dbpedia";
    const names = ["MPNet", "Qwen3-Embedding", "CodeBERT"];
    const colors = () => [cv("--qa"), cv("--qb"), cv("--qc")];
    function draw() {
      clear(svg);
      const ti = +slider.value;
      $("#thrVal").textContent = P.thresholds[ti].toFixed(2);
      const W = 480, H = 300, m = { l: 46, r: 14, t: 14, b: 40 };
      const X = (v) => m.l + v * (W - m.l - m.r), Y = (v) => H - m.b - v * (H - m.t - m.b);
      s("rect", { x: X(0), y: Y(1), width: X(1) - X(0), height: Y(0.9) - Y(1), fill: cv("--good-soft") }, svg);
      for (let i = 0; i <= 5; i++) {
        const v = i / 5;
        s("line", { x1: X(0), x2: X(1), y1: Y(v), y2: Y(v), stroke: cv("--line"), "stroke-width": 0.6 }, svg);
        s("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end", "font-size": 10, text: v.toFixed(1) }, svg);
        s("text", { x: X(v), y: H - m.b + 15, "text-anchor": "middle", "font-size": 10, text: v.toFixed(1) }, svg);
      }
      s("text", { x: X(0.99), y: Y(0.95) + 4, "text-anchor": "end", "font-size": 10, text: "reliable zone: accuracy ≥ 0.9" }, svg).style.fill = cv("--good");
      s("text", { x: (X(0) + X(1)) / 2, y: H - 6, "text-anchor": "middle", "font-size": 11, text: "hit rate (share of queries answered from cache)" }, svg);
      s("text", { x: 12, y: (Y(0) + Y(1)) / 2, "text-anchor": "middle", "font-size": 11, transform: `rotate(-90 12 ${(Y(0) + Y(1)) / 2})`, text: "hit accuracy" }, svg);
      const col = colors();
      const rows = [];
      names.forEach((n, k) => {
        const pts = P.riskCoverage[g][n];
        s("polyline", { points: pts.map(([shr, risk]) => `${X(shr)},${Y(1 - risk)}`).join(" "), fill: "none", stroke: col[k], "stroke-width": 2, opacity: 0.8 }, svg);
        pts.forEach(([shr, risk]) => s("circle", { cx: X(shr), cy: Y(1 - risk), r: 2.4, fill: col[k] }, svg));
        const [shr, risk] = pts[ti];
        s("circle", { cx: X(shr), cy: Y(1 - risk), r: 7, fill: col[k], stroke: cv("--surface"), "stroke-width": 2 }, svg);
        rows.push([n, shr, 1 - risk, shr * (1 - risk), col[k]]);
      });
      $("#rcTable").innerHTML = `<table class="mini"><thead><tr><th>fine-tuned encoder</th><th>hit rate</th><th>hit accuracy</th><th>correct hit rate</th></tr></thead><tbody>` +
        rows.map(([n, shr, acc, chr, c]) => `<tr><td><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${c};margin-right:6px"></span>${n}</td><td>${fmt(shr)}</td><td style="color:${acc >= 0.9 ? "var(--good)" : "inherit"};font-weight:${acc >= 0.9 ? 700 : 400}">${fmt(acc)}</td><td>${fmt(chr)}</td></tr>`).join("") +
        `</tbody></table><p class="note">Raising θ trades coverage for reliability. On DBpedia high θ reaches the reliable zone; on Wikidata most encoders never do (CodeBERT even gets <i>less</i> accurate as θ rises). Test streams, K = 128, pooled over 5 seeds × 5 orders.</p>`;
    }
    slider.addEventListener("input", draw);
    segment("rcSeg", (d) => { g = d.g; draw(); });
    onTheme(draw);
  })();

  /* ------------------------------------------------------------------ 5. comparison + dataset */
  (function novelty() {
    if (!$("#cmpTable")) return;
    const Y = '<span class="yes">✓</span>', N = '<span class="no">✗</span>', Pp = (t) => `<span class="part">${t}</span>`;
    const rows = [
      ["Query2Vec, Querc (SQL workloads)", Y, N, N, Y],
      ["SPBERT (pretrained on SPARQL logs)", Y, N, N, Y],
      ["Text and code encoders (E5, BGE, CodeBERT, …)", Y, N, N, Y],
      ["Containment / equivalence checking", N, N, Pp("holds on every graph"), Pp("yes, but costly")],
      ["SQA (execution-based query assessment)", N, Pp("partial"), Y, N],
      ["Embedding-based semantic caches (GPTCache-style)", Y, N, N, Y],
      ["Structural SPARQL caches, template prefetching", N, N, N, Y],
      ["<b>Results-aware embeddings + SQuARE (ours)</b>", Y, Y, Y, Y],
    ];
    $("#cmpTable").innerHTML = `<thead><tr><th>Approach</th><th>Learns a query<br>embedding</th><th>Trained on<br>query answers</th><th>Relatedness on<br>the target graph</th><th>Query text only<br>at inference</th></tr></thead><tbody>` +
      rows.map((r, i) => `<tr class="${i === rows.length - 1 ? "ours" : ""}">${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("") + "</tbody>";
    const d = P.dataset;
    const n = (x) => x.toLocaleString();
    $("#dsTable").innerHTML = `<thead><tr><th></th><th>DBpedia</th><th>Wikidata</th></tr></thead><tbody>
      <tr><td>Unique queries</td><td>${n(d.dbpedia.queries)}</td><td>${n(d.wikidata.queries)}</td></tr>
      <tr><td>Queries (train / valid / test)</td><td>${d.dbpedia.split}</td><td>${d.wikidata.split}</td></tr>
      <tr><td>Positive pairs (J ≥ 0.8)</td><td>${n(d.dbpedia.pos)}</td><td>${n(d.wikidata.pos)}</td></tr>
      <tr><td>Negative pairs (J ≤ 0.1)</td><td>${n(d.dbpedia.neg)}</td><td>${n(d.wikidata.neg)}</td></tr>
      <tr><td>Positives with J = 1</td><td>${d.dbpedia.j1}</td><td>${d.wikidata.j1}</td></tr>
      <tr><td>Query templates</td><td>${n(d.dbpedia.templates)}</td><td>${n(d.wikidata.templates)}</td></tr>
      <tr><td>Test anchors with a training template</td><td>${d.dbpedia.seen}</td><td>${d.wikidata.seen}</td></tr></tbody>`;
  })();
})();
