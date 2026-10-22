// SafariSeg: YOLO26n-seg (polygon) ทำงานบนเบราว์เซอร์ด้วย onnxruntime-web
const MODEL_URL = "model/best.onnx";
const META_URL = "model/meta.json";
const COLORS = ["#3b82f6", "#111111", "#e0a100"];
const MASK_THRESHOLD = 0.5;
const IOU_THRESHOLD = 0.6;

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

let session, meta, IMG = 640;
let current = null;   // { img, raw, lb, ms }

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

// ---------- โหลดโมเดล ----------
async function init() {
  setBusy("กำลังโหลดโมเดล…");
  try {
    meta = await (await fetch(META_URL)).json();
    IMG = meta.imgsz || 640;
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
    $("status").textContent = `โมเดลพร้อม · ${meta.names.length} labels · ขนาด ${meta.size_mb} MB`;
    buildSamples();
    renderResults();
    if (new URLSearchParams(location.search).has("auto")) $("samples").querySelector("button")?.click();
  } catch (e) {
    console.error(e);
    $("status").textContent = "โหลดโมเดลไม่สำเร็จ: " + e.message;
  }
  setBusy(null);
}

function setBusy(text) {
  $("busy").hidden = !text;
  if (text) $("busyText").textContent = text;
}

// ---------- รับภาพ ----------
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

async function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const url = URL.createObjectURL(file);
  document.querySelectorAll(".samples button").forEach((b) => b.classList.remove("on"));
  await runOn(await loadImage(url));
  URL.revokeObjectURL(url);
}

async function runOn(img) {
  if (!session) return;
  setBusy("กำลังทำนาย…");
  await new Promise((r) => setTimeout(r, 20));   // ให้ UI วาด spinner ก่อน
  const { tensor, lb } = preprocess(img);
  const t0 = performance.now();
  const out = await session.run({ [session.inputNames[0]]: tensor });
  const ms = performance.now() - t0;
  current = { img, raw: out, lb, ms };
  setBusy(null);
  render();
}

// ---------- Letterbox: ย่อคงสัดส่วนให้ด้านยาว = 640 แล้วเติมขอบสีเทา ----------
function preprocess(img) {
  const w = img.naturalWidth, h = img.naturalHeight;
  const r = Math.min(IMG / w, IMG / h);
  const nw = Math.round(w * r), nh = Math.round(h * r);
  const px = Math.floor((IMG - nw) / 2), py = Math.floor((IMG - nh) / 2);

  const c = document.createElement("canvas");
  c.width = IMG; c.height = IMG;
  const g = c.getContext("2d");
  g.fillStyle = "rgb(114,114,114)";
  g.fillRect(0, 0, IMG, IMG);
  g.drawImage(img, px, py, nw, nh);
  const data = g.getImageData(0, 0, IMG, IMG).data;

  // HWC RGBA -> CHW RGB, 0..1
  const area = IMG * IMG;
  const f = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    f[i] = data[i * 4] / 255;
    f[i + area] = data[i * 4 + 1] / 255;
    f[i + 2 * area] = data[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", f, [1, 3, IMG, IMG]), lb: { r, px, py, w, h } };
}

// ---------- Decode output ----------
function decode(out, conf) {
  const tensors = Object.values(out);
  const proto = tensors.find((t) => t.dims.length === 4);
  const pred = tensors.find((t) => t.dims.length === 3);
  const nc = meta.names.length, nm = proto.dims[1];
  const p = pred.data, dets = [];

  if (pred.dims[2] === 6 + nm) {
    // แบบ end-to-end (YOLO26 ไม่ต้องทำ NMS): [1, N, x1 y1 x2 y2 score cls mask...]
    const n = pred.dims[1], s = pred.dims[2];
    for (let i = 0; i < n; i++) {
      const o = i * s, score = p[o + 4];
      if (score < conf) continue;
      dets.push({ box: [p[o], p[o + 1], p[o + 2], p[o + 3]], score, cls: Math.round(p[o + 5]),
                  coef: p.slice(o + 6, o + 6 + nm) });
    }
    return dets;
  }

  // แบบเดิม: [1, 4+nc+nm, N] ต้องทำ NMS เอง
  const ch = pred.dims[1], n = pred.dims[2];
  for (let i = 0; i < n; i++) {
    let best = 0, cls = 0;
    for (let c = 0; c < nc; c++) {
      const v = p[(4 + c) * n + i];
      if (v > best) { best = v; cls = c; }
    }
    if (best < conf) continue;
    const cx = p[i], cy = p[n + i], w = p[2 * n + i], h = p[3 * n + i];
    const coef = new Float32Array(nm);
    for (let k = 0; k < nm; k++) coef[k] = p[(4 + nc + k) * n + i];
    dets.push({ box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], score: best, cls, coef });
  }
  return nms(dets);
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return ua > 0 ? inter / ua : 0;
}

function nms(dets) {
  dets.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const d of dets) {
    if (keep.every((k) => k.cls !== d.cls || iou(k.box, d.box) < IOU_THRESHOLD)) keep.push(d);
    if (keep.length >= 100) break;
  }
  return keep;
}

// ---------- Mask -> Polygon ----------
// mask = sigmoid(coef · proto) ที่ความละเอียด 160x160 แล้วสุ่มแบบ bilinear ขึ้นเป็น 640 เฉพาะในกรอบ
function maskPolygon(det, proto) {
  const [, nm, mh, mw] = proto.dims;
  const P = proto.data, sx = mw / IMG, sy = mh / IMG;
  const x1 = Math.max(0, Math.floor(det.box[0])), y1 = Math.max(0, Math.floor(det.box[1]));
  const x2 = Math.min(IMG, Math.ceil(det.box[2])), y2 = Math.min(IMG, Math.ceil(det.box[3]));
  const bw = x2 - x1, bh = y2 - y1;
  if (bw < 2 || bh < 2) return null;

  // logit ที่ความละเอียด proto เฉพาะช่วงที่ครอบกรอบ (+1 ช่องสำหรับ interpolate)
  const gx1 = Math.max(0, Math.floor(x1 * sx) - 1), gy1 = Math.max(0, Math.floor(y1 * sy) - 1);
  const gx2 = Math.min(mw - 1, Math.ceil(x2 * sx) + 1), gy2 = Math.min(mh - 1, Math.ceil(y2 * sy) + 1);
  const gw = gx2 - gx1 + 1, gh = gy2 - gy1 + 1, area = mh * mw;
  const logit = new Float32Array(gw * gh);
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      const idx = (gy1 + y) * mw + gx1 + x;
      let s = 0;
      for (let k = 0; k < nm; k++) s += det.coef[k] * P[k * area + idx];
      logit[y * gw + x] = s;
    }
  }

  // bilinear ขึ้นเป็นพิกัด 640 (sigmoid > 0.5 เท่ากับ logit > 0)
  const T = Math.log(MASK_THRESHOLD / (1 - MASK_THRESHOLD));
  const m = new Uint8Array(bw * bh);
  let count = 0;
  for (let y = 0; y < bh; y++) {
    const fy = Math.min(gh - 1.001, Math.max(0, (y1 + y + 0.5) * sy - 0.5 - gy1));
    const y0 = Math.floor(fy), dy = fy - y0;
    for (let x = 0; x < bw; x++) {
      const fx = Math.min(gw - 1.001, Math.max(0, (x1 + x + 0.5) * sx - 0.5 - gx1));
      const x0 = Math.floor(fx), dx = fx - x0;
      const a = logit[y0 * gw + x0], b = logit[y0 * gw + x0 + 1];
      const c = logit[(y0 + 1) * gw + x0], d = logit[(y0 + 1) * gw + x0 + 1];
      const v = (a * (1 - dx) + b * dx) * (1 - dy) + (c * (1 - dx) + d * dx) * dy;
      if (v > T) { m[y * bw + x] = 1; count++; }
    }
  }
  if (count < 20) return null;

  const contour = traceLargest(m, bw, bh);
  if (!contour || contour.length < 3) return null;
  const poly = simplify(contour, 1.5).map(([x, y]) => [x + x1, y + y1]);
  return { poly, pixels: count };
}

// หา component ที่ใหญ่ที่สุด แล้วเดินตามขอบด้วย Moore-neighbor tracing
function traceLargest(m, w, h) {
  const lab = new Int32Array(w * h);
  let bestId = 0, bestSize = 0, bestStart = -1, id = 0;
  const stack = [];
  for (let i = 0; i < w * h; i++) {
    if (!m[i] || lab[i]) continue;
    id++;
    let size = 0;
    stack.push(i); lab[i] = id;
    while (stack.length) {
      const j = stack.pop(); size++;
      const x = j % w, y = (j / w) | 0;
      if (x > 0 && m[j - 1] && !lab[j - 1]) { lab[j - 1] = id; stack.push(j - 1); }
      if (x < w - 1 && m[j + 1] && !lab[j + 1]) { lab[j + 1] = id; stack.push(j + 1); }
      if (y > 0 && m[j - w] && !lab[j - w]) { lab[j - w] = id; stack.push(j - w); }
      if (y < h - 1 && m[j + w] && !lab[j + w]) { lab[j + w] = id; stack.push(j + w); }
    }
    if (size > bestSize) { bestSize = size; bestId = id; bestStart = i; }
  }
  if (bestStart < 0) return null;

  const inside = (x, y) => x >= 0 && y >= 0 && x < w && y < h && lab[y * w + x] === bestId;
  const dirs = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
  const sx = bestStart % w, sy = (bestStart / w) | 0;
  const pts = [[sx, sy]];
  let x = sx, y = sy, dir = 7;
  for (let guard = 0; guard < 4 * (w + h) * 4; guard++) {
    let found = false;
    for (let k = 0; k < 8; k++) {
      const d = (dir + 6 + k) % 8;   // เริ่มมองจากด้านซ้ายของทิศเดิม
      const nx = x + dirs[d][0], ny = y + dirs[d][1];
      if (inside(nx, ny)) { x = nx; y = ny; dir = d; found = true; break; }
    }
    if (!found || (x === sx && y === sy)) break;
    pts.push([x, y]);
  }
  return pts;
}

// Douglas–Peucker ลดจำนวนจุดของ polygon
function simplify(pts, eps) {
  if (pts.length < 4) return pts;
  const keep = new Uint8Array(pts.length);
  keep[0] = keep[pts.length - 1] = 1;
  const stack = [[0, pts.length - 1]];
  while (stack.length) {
    const [a, b] = stack.pop();
    const [ax, ay] = pts[a], [bx, by] = pts[b];
    const dx = bx - ax, dy = by - ay, len = Math.hypot(dx, dy) || 1;
    let maxD = 0, idx = -1;
    for (let i = a + 1; i < b; i++) {
      const d = Math.abs(dy * pts[i][0] - dx * pts[i][1] + bx * ay - by * ax) / len;
      if (d > maxD) { maxD = d; idx = i; }
    }
    if (maxD > eps) { keep[idx] = 1; stack.push([a, idx], [idx, b]); }
  }
  return pts.filter((_, i) => keep[i]);
}

// ---------- วาดผล ----------
function render() {
  if (!current) return;
  const { img, raw, lb, ms } = current;
  const conf = +$("conf").value;
  const dets = decode(raw, conf);
  const proto = Object.values(raw).find((t) => t.dims.length === 4);

  // พิกัด 640 -> พิกัดภาพจริง
  const toImg = ([x, y]) => [(x - lb.px) / lb.r, (y - lb.py) / lb.r];
  const imgArea = lb.w * lb.h;
  for (const d of dets) {
    const mp = maskPolygon(d, proto);
    d.poly = mp ? mp.poly.map(toImg) : null;
    d.areaPct = mp ? (mp.pixels / (lb.r * lb.r)) / imgArea * 100 : 0;
    const [a, b] = [toImg([d.box[0], d.box[1]]), toImg([d.box[2], d.box[3]])];
    d.ibox = [Math.max(0, a[0]), Math.max(0, a[1]), Math.min(lb.w, b[0]), Math.min(lb.h, b[1])];
  }

  // ขนาด canvas = ภาพจริง (จำกัดด้านยาว 1600 เพื่อความเร็ว)
  const k = Math.min(1, 1600 / Math.max(lb.w, lb.h));
  canvas.width = Math.round(lb.w * k);
  canvas.height = Math.round(lb.h * k);
  ctx.setTransform(k, 0, 0, k, 0, 0);
  ctx.drawImage(img, 0, 0, lb.w, lb.h);

  const lw = Math.max(2, Math.max(lb.w, lb.h) / 400);
  const showMask = $("showMask").checked, showBox = $("showBox").checked, showLabel = $("showLabel").checked;
  for (const d of dets) {
    const col = COLORS[d.cls % COLORS.length];
    if (d.poly && showMask) {
      ctx.beginPath();
      d.poly.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.fillStyle = col + "55";
      ctx.fill();
      ctx.lineWidth = lw;
      ctx.strokeStyle = col;
      ctx.stroke();
      ctx.fillStyle = "#fff";
      for (const [x, y] of d.poly) { ctx.beginPath(); ctx.arc(x, y, lw * 0.9, 0, 7); ctx.fill(); }
    }
    if (showBox || !d.poly) {
      ctx.setLineDash([lw * 3, lw * 2]);
      ctx.lineWidth = lw;
      ctx.strokeStyle = col;
      ctx.strokeRect(d.ibox[0], d.ibox[1], d.ibox[2] - d.ibox[0], d.ibox[3] - d.ibox[1]);
      ctx.setLineDash([]);
    }
  }
  if (showLabel) {
    const fs = Math.max(13, Math.max(lb.w, lb.h) / 45);
    ctx.font = `600 ${fs}px "IBM Plex Sans Thai", sans-serif`;
    for (const d of dets) {
      const txt = `${meta.names[d.cls]} ${(d.score * 100).toFixed(0)}%`;
      const tw = ctx.measureText(txt).width + fs * 0.6;
      const x = d.ibox[0], y = Math.max(fs * 1.4, d.ibox[1]);
      ctx.fillStyle = COLORS[d.cls % COLORS.length];
      ctx.fillRect(x, y - fs * 1.4, tw, fs * 1.4);
      ctx.fillStyle = d.cls === 2 ? "#111" : "#fff";
      ctx.fillText(txt, x + fs * 0.3, y - fs * 0.35);
    }
  }
  canvas.hidden = false;
  $("placeholder").hidden = true;
  updateSide(dets, ms);
}

function updateSide(dets, ms) {
  $("nObj").textContent = dets.length;
  $("tInf").textContent = `${ms.toFixed(0)} ms`;
  const counts = meta.names.map((n, i) => [n, i, dets.filter((d) => d.cls === i).length]).filter((c) => c[2]);
  $("counts").innerHTML = counts.map(([n, i, c]) =>
    `<span style="background:${COLORS[i]};${i === 2 ? "color:#111" : ""}">${n} × ${c}</span>`).join("");
  $("list").innerHTML = dets.map((d, i) => `<li><i style="background:${COLORS[d.cls]}"></i>
    <span>#${i + 1} ${meta.names[d.cls]} · ${(d.score * 100).toFixed(1)}%</span>
    <small>${d.poly ? d.poly.length + " จุด · " : ""}${d.areaPct.toFixed(1)}% ของภาพ</small></li>`).join("");
  $("status").textContent = dets.length ? "คลิกตัวเลือกด้านบนเพื่อเปลี่ยนการแสดงผล" : "ไม่พบวัตถุ ลองลดค่า Confidence";
}

// ---------- ภาพตัวอย่าง ----------
function buildSamples() {
  const box = $("samples");
  box.innerHTML = "";
  for (const s of meta.samples || []) {
    const b = document.createElement("button");
    b.innerHTML = `<img src="samples/${s}" alt="sample" loading="lazy">`;
    b.onclick = async () => {
      document.querySelectorAll(".samples button").forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      await runOn(await loadImage(`samples/${s}`));
    };
    box.appendChild(b);
  }
}

// ---------- ส่วนผลการเทรน ----------
function renderResults() {
  const r = meta.results;
  if (!r) return;
  const f = (v) => (v == null ? "–" : v.toFixed(3));
  $("metricCards").innerHTML = [
    ["Rect", "Box mAP50", r.rect.map50], ["Rect", "Box mAP50-95", r.rect.map],
    ["Polygon", "Mask mAP50", r.polygon.mask_map50], ["Polygon", "Mask mAP50-95", r.polygon.mask_map],
  ].map(([t, k, v]) => `<div class="card"><span class="tag">${t}</span><small>${k} (test)</small><b>${f(v)}</b></div>`).join("");

  const total = { images: 0 };
  meta.names.forEach((n) => (total[n] = 0));
  for (const c of Object.values(meta.counts)) for (const k in total) total[k] += c[k];
  const row = (s, c, tag = "td") => `<tr><${tag}>${s}</${tag}><${tag}>${c.images}</${tag}>` +
    meta.names.map((n) => `<${tag}>${c[n]}</${tag}>`).join("") +
    `<${tag}>${meta.names.reduce((a, n) => a + c[n], 0)}</${tag}></tr>`;
  $("objTable").querySelector("tbody").innerHTML =
    Object.entries(meta.counts).map(([s, c]) => row(s, c)).join("") + row("รวม", total, "th");
  const tr = meta.counts.train;
  $("countNote").textContent = `ใช้ภาพทั้งหมด ${total.images} ภาพ (train ${tr.images} / val ${meta.counts.val.images} / test ${meta.counts.test.images}) ` +
    `ชุด train มี object ต่อ label ${meta.names.map((n) => `${n} ${tr[n]}`).join(", ")} ทุก label มากกว่า 100 ตามเกณฑ์`;

  $("clsTable").querySelector("tbody").innerHTML = meta.names.map((n, i) =>
    `<tr><td>${n}</td><td>${f(r.rect.per_class[i].map50)}</td><td>${f(r.rect.per_class[i].map)}</td>
     <td>${f(r.polygon.per_class[i].box_map50)}</td><td>${f(r.polygon.per_class[i].mask_map50)}</td>
     <td>${f(r.polygon.per_class[i].mask_map)}</td></tr>`).join("");
}

// ---------- events ----------
$("file").onchange = (e) => handleFile(e.target.files[0]);
$("camera").onchange = (e) => handleFile(e.target.files[0]);
$("conf").oninput = (e) => { $("confVal").textContent = (+e.target.value).toFixed(2); render(); };
["showMask", "showBox", "showLabel"].forEach((id) => ($(id).onchange = render));
const drop = $("drop");
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); handleFile(e.dataTransfer.files[0]); };

init();
