// Latihan wawancara: deteksi wajah di browser (MediaPipe), klasifikasi emosi di server lokal,
// ringkasan dihitung sepenuhnya di browser. Tidak ada data sesi yang disimpan di server.
// Tambahkan ?demo=1 pada alamat untuk mode demo (kamera dan hasil simulasi, tanpa model).

const DEMO = new URLSearchParams(location.search).has('demo');

const CONFIG = {
  selfHost: false,   // true: pakai berkas di static/mediapipe/ (tanpa CDN pihak ketiga)
  mp: { cdn: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14',
        model: 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite' },
  cropSize: 224, marginRatio: 0.10, jpegQuality: 0.9,
  minIntervalMs: 150, maxInFlight: 2, requestTimeoutMs: 2500,
  faceStaleMs: 500, smoothTauMs: 40, minFaceScore: 0.5,
  binMs: 3000
};
const MP_URLS = CONFIG.selfHost
  ? { bundle: '/static/mediapipe/vision_bundle.mjs', wasm: '/static/mediapipe/wasm', model: '/static/mediapipe/blaze_face_short_range.tflite' }
  : { bundle: CONFIG.mp.cdn + '/vision_bundle.mjs', wasm: CONFIG.mp.cdn + '/wasm', model: CONFIG.mp.model };

// Urutan sama dengan keluaran server (InterviewEmotionPredictor.target_classes).
const CLASSES = [
  { key: 'Happy',   id: 'senang', label: 'Senang', desc: 'Senyum jelas atau wajah cerah' },
  { key: 'Neutral', id: 'netral', label: 'Netral', desc: 'Wajah datar atau sedang fokus' },
  { key: 'Tense',   id: 'tegang', label: 'Tegang', desc: 'Alis atau wajah tampak menegang' },
  { key: 'Calm',    id: 'tenang', label: 'Tenang', desc: 'Wajah rileks, senyum tipis' }
];
const K = CLASSES.length;

const QUESTIONS = [
  {
    text: 'Ceritakan tentang diri Anda secara singkat.',
    tip: 'Fokus pada latar belakang profesional/akademis, pencapaian utama, dan alasan Anda tertarik pada posisi ini.'
  },
  {
    text: 'Mengapa Anda tertarik melamar posisi ini di perusahaan kami?',
    tip: 'Tunjukkan bahwa Anda telah meriset perusahaan dan jelaskan bagaimana posisi ini selaras dengan nilai serta tujuan karier Anda.'
  },
  {
    text: 'Apa kelebihan utama Anda yang paling relevan untuk peran ini?',
    tip: 'Pilih 1–2 keterampilan utama, lalu sertakan contoh nyata atau hasil konkret dari pengalaman Anda sebelumnya.'
  },
  {
    text: 'Apa kekurangan terbesar Anda, dan bagaimana cara Anda mengatasinya?',
    tip: 'Jelaskan area pengembangan diri secara jujur, lalu tekankan langkah konkret yang sedang Anda lakukan untuk memperbaikinya.'
  },
  {
    text: 'Ceritakan pengalaman saat Anda menghadapi tekanan atau tantangan berat.',
    tip: 'Gunakan metode STAR (Situation, Task, Action, Result) untuk menjelaskan situasi, tindakan nyata Anda, dan hasil akhirnya.'
  },
  {
    text: 'Pencapaian apa yang paling Anda banggakan dalam studi atau karier Anda?',
    tip: 'Soroti proyek atau momen spesifik di mana kontribusi langsung Anda memberikan dampak nyata atau nilai tambah.'
  },
  {
    text: 'Ceritakan pengalaman saat Anda membuat kesalahan dan pelajaran yang Anda dapatkan.',
    tip: 'Fokus pada tanggung jawab yang Anda ambil, solusi yang Anda jalankan, dan perbaikan pola kerja setelah kejadian tersebut.'
  },
  {
    text: 'Ceritakan pengalaman saat Anda berbeda pendapat dengan rekan kerja atau atasan.',
    tip: 'Tunjukkan kemampuan komunikasi yang tenang, keterbukaan menerima sudut pandang lain, dan bagaimana kesepakatan dicapai secara profesional.'
  },
  {
    text: 'Bagaimana cara Anda mengelola prioritas saat memiliki banyak tugas dengan tenggat waktu berdekatan?',
    tip: 'Jelaskan metode manajemen waktu yang Anda gunakan, seperti skala prioritas, alokasi fokus, atau komunikasi tenggat waktu secara transparan.'
  },
  {
    text: 'Mengapa kami harus memilih Anda dibandingkan kandidat lainnya?',
    tip: 'Rangkum kombinasi unik antara keterampilan teknis, soft skill, dan antusiasme tinggi Anda untuk memberikan dampak bagi tim.'
  },
  {
    text: 'Lingkungan kerja seperti apa yang membuat Anda dapat bekerja secara optimal?',
    tip: 'Sebutkan budaya kerja yang Anda apresiasi (misalnya kolaboratif, komunikatif, atau terstruktur) secara positif.'
  },
  {
    text: 'Di mana Anda melihat diri Anda dalam lima tahun ke depan?',
    tip: 'Hubungkan dengan pengembangan keterampilan profesional, pertumbuhan tanggung jawab, dan kontribusi jangka panjang yang ingin Anda capai.'
  },
  {
    text: 'Bagaimana cara Anda menyikapi umpan balik atau kritik terhadap hasil kerja Anda?',
    tip: 'Tunjukkan sikap terbuka, tidak defensif, dan keinginan untuk memanfaatkan kritik sebagai bahan evaluasi serta pengembangan diri.'
  },
  {
    text: 'Mengapa Anda memutuskan untuk mencari peluang kerja baru saat ini?',
    tip: 'Fokus pada dorongan positif untuk tumbuh dan mencari tantangan baru. Hindari membicarakan hal negatif tentang tempat belajar/kerja lama.'
  },
  {
    text: 'Apakah ada pertanyaan yang ingin Anda ajukan kepada kami?',
    tip: 'Ajukan 1–2 pertanyaan berkualitas tentang dinamika tim, tantangan utama pada peran ini, atau ekspektasi perusahaan dalam 3 bulan pertama.'
  }
];

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const el = {
  root: document.documentElement, layout: $('#layout'), cam: $('#cam'), video: $('#video'),
  overlay: $('#overlay'), demoCanvas: $('#demoCanvas'), camPh: $('#camPh'), camPhText: $('#camPhText'),
  camSpinner: $('#camSpinner'), camBtn: $('#camBtn'), camMsg: $('#camMsg'), checks: $('#checks'),
  startBtn: $('#startBtn'), startHint: $('#startHint'), consent: $('#consent'),
  meter: $('#meter'), meterNow: $('#meterNow'), meterRows: $('#meterRows'), previewBtn: $('#previewBtn'),
  progressText: $('#progressText'), timer: $('#timer'), timerLabel: $('#timerLabel'), timerValue: $('#timerValue'),
  qBar: $('#qBar'), qBarFill: $('#qBarFill'), questionText: $('#questionText'), questionTip: $('#questionTip'),
  timeNotice: $('#timeNotice'), pauseBtn: $('#pauseBtn'), pauseLabel: $('#pauseLabel'), nextBtn: $('#nextBtn'), endBtn: $('#endBtn'),
  reportMeta: $('#reportMeta'), reportBody: $('#reportBody'), sr: $('#sr'), techOut: $('#techOut'),
  prefsDlg: $('#prefsDlg'), endDlg: $('#endDlg')
};

const state = {
  screen: 'setup',
  cam: { active: false, loading: false, loadingText: '', error: '' },
  previewHidden: false,
  session: null, report: null
};

const selectedPreview = $('input[name="preview"]:checked');
if (selectedPreview) {
  state.previewHidden = selectedPreview.value === 'hide';
}

$$('input[name="preview"]').forEach(radio => {
  radio.addEventListener('change', (e) => {
    state.previewHidden = e.target.value === 'hide';
    renderCamPh();
  });
});

/* ---------------------------------------------------------------- util */
const fmtClock = ms => { const s = Math.max(0, Math.round(ms / 1000)); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); };
const fmtLong = ms => { const s = Math.round(ms / 1000), m = Math.floor(s / 60), r = s % 60;
  return m ? `${m} menit${r ? ' ' + r + ' detik' : ''}` : `${r} detik`; };
const argmax = a => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
function roundShares(vals) {            // persen bulat yang jumlahnya tepat 100 (largest remainder)
  const sum = vals.reduce((a, b) => a + b, 0);
  if (!sum) return vals.map(() => 0);
  const raw = vals.map(v => v / sum * 100), fl = raw.map(Math.floor);
  let left = 100 - fl.reduce((a, b) => a + b, 0);
  raw.map((v, i) => [v - fl[i], i]).sort((a, b) => b[0] - a[0]).forEach(([, i]) => { if (left-- > 0) fl[i]++; });
  return fl;
}
const announce = msg => { el.sr.textContent = ''; setTimeout(() => { el.sr.textContent = msg; }, 50); };
const safe = fn => { try { return fn(); } catch { return null; } };

/* ---------------------------------------------------------------- preferensi tampilan */
function applyPrefs(p) {
  el.root.dataset.theme = p.theme; el.root.dataset.size = p.size;
  if (p.motion === 'reduce') el.root.dataset.motion = 'reduce'; else delete el.root.dataset.motion;
}
function loadPrefs() {
  const d = { theme: 'auto', size: 'm', motion: 'auto' };
  const saved = safe(() => JSON.parse(localStorage.getItem('lw.prefs') || '{}')) || {};
  const p = { ...d, ...saved };
  for (const k of Object.keys(d)) { const r = $(`input[name=${k}][value="${p[k]}"]`); if (r) r.checked = true; }
  applyPrefs(p);
}
function readPrefs() { const g = n => $(`input[name=${n}]:checked`).value; return { theme: g('theme'), size: g('size'), motion: g('motion') }; }
$$('#prefsDlg input').forEach(i => i.addEventListener('change', () => {
  const p = readPrefs(); applyPrefs(p); safe(() => localStorage.setItem('lw.prefs', JSON.stringify(p)));
}));
$('#openPrefs').addEventListener('click', () => el.prefsDlg.showModal());

/* ---------------------------------------------------------------- navigasi layar */
const TITLES = { setup: 'Latihan wawancara', session: 'Menjawab pertanyaan - Latihan wawancara', report: 'Ringkasan sesi - Latihan wawancara' };
function go(name, focus = true) {
  state.screen = name;
  $('#screen-setup').hidden = name !== 'setup';
  $('#screen-session').hidden = name !== 'session';
  $('#screen-report').hidden = name !== 'report';
  el.layout.hidden = name === 'report';
  el.layout.dataset.screen = name;
  el.meter.hidden = name !== 'session';
  el.checks.hidden = name === 'session';
  $$('#steps li').forEach(li => { if (li.dataset.step === name) li.setAttribute('aria-current', 'step'); else li.removeAttribute('aria-current'); });
  document.title = TITLES[name];
  const h = { setup: '#h-setup', session: '#questionText', report: '#h-report' }[name];
  window.scrollTo(0, 0);
  if (focus) $(h).focus({ preventScroll: true });
  renderCamPh();
}

/* ---------------------------------------------------------------- kamera + deteksi */
let detector = null, stream = null, rafId = 0, lastVT = -1;
let targetBox = null, drawBox = null, faceAt = 0;
const metrics = { det: 0, detFps: 0, ping: 0, rtt: 0, server: 0, infer: 0, wait: 0, kb: 0 };
let accent = '#0e5e5c';

const faceVisible = () => targetBox && performance.now() - faceAt < CONFIG.faceStaleMs;

async function loadDetector() {
  if (detector) return;
  let mp;
  try {
    mp = await import(MP_URLS.bundle);
    const vision = await mp.FilesetResolver.forVisionTasks(MP_URLS.wasm);
    for (const delegate of ['GPU', 'CPU']) {
      try {
        detector = await mp.FaceDetector.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MP_URLS.model, delegate }, runningMode: 'VIDEO',
          minDetectionConfidence: CONFIG.minFaceScore });
        return;
      } catch (e) { console.warn('Delegate gagal:', delegate, e); }
    }
    throw new Error('FaceDetector gagal dibuat');
  } catch (e) { console.error(e); const err = new Error('detector'); err.cause = e; throw err; }
}

function renderCamPh() {
  const c = state.cam, s = state.session;
  const paused = !!(s && s.paused && state.screen === 'session');
  const hidden = c.active && state.previewHidden;
  let st = 'off', text = 'Kamera belum aktif.', btn = true, label = 'Aktifkan kamera', spin = false;
  if (c.loading) { st = 'loading'; text = c.loadingText; btn = false; spin = true; }
  else if (c.error) { st = 'error'; text = c.error; label = 'Coba lagi'; }
  else if (c.active) {
    st = 'live'; btn = false;
    if (hidden) text = 'Pratinjau disembunyikan. Kamera tetap membaca ekspresi.';
    else if (paused) text = 'Sesi dijeda.';
  }
  el.cam.dataset.state = st; el.cam.dataset.hidden = String(hidden); el.cam.dataset.paused = String(paused);
  el.camPhText.textContent = text;
  el.camSpinner.hidden = !spin;
  el.camBtn.hidden = !btn;
  el.camBtn.lastChild.textContent = ' ' + label;
  el.previewBtn.textContent = state.previewHidden ? 'Tampilkan pratinjau' : 'Sembunyikan pratinjau';
  $('#chkCam').dataset.ok = String(c.active);
  $('#chkCamText').textContent = c.active ? (DEMO ? 'Kamera simulasi aktif' : 'Kamera aktif') : 'Kamera belum aktif';
  updateStartBtn();
}

function sizeOverlay() {
  const w = DEMO ? 640 : el.video.videoWidth, h = DEMO ? 360 : el.video.videoHeight;
  el.overlay.width = w; el.overlay.height = h;
  el.cam.style.setProperty('--ar', `${w} / ${h}`); el.cam.style.setProperty('--arn', String(w / h));
}

async function enableCamera() {
  const c = state.cam;
  if (c.loading || c.active) return;
  c.error = ''; c.loading = true; c.loadingText = DEMO ? 'Menyiapkan kamera simulasi...' : 'Memuat model deteksi wajah...';
  renderCamPh();
  try {
    if (DEMO) {
      el.cam.dataset.demo = 'true'; el.demoCanvas.hidden = false;
    } else {
      if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error('insecure');
      await loadDetector();
      c.loadingText = 'Meminta izin kamera...'; renderCamPh();
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      el.video.srcObject = stream;
      if (el.video.readyState < 1) await new Promise(r => el.video.addEventListener('loadedmetadata', r, { once: true }));
      await el.video.play();
    }
    sizeOverlay();
    c.active = true; c.loading = false;
    el.camMsg.textContent = '';
    startLoops();
  } catch (e) {
    c.loading = false; c.active = false;
    const n = e && e.name, m = e && e.message;
    c.error = m === 'insecure' ? 'Kamera hanya bisa dipakai lewat alamat https://. Buka halaman ini dengan https://.'
      : m === 'detector' ? 'Model deteksi wajah gagal dimuat. Periksa koneksi internet, lalu coba lagi.'
      : n === 'NotAllowedError' ? 'Izin kamera ditolak. Pilih ikon gembok di bilah alamat, izinkan kamera, lalu coba lagi.'
      : n === 'NotFoundError' ? 'Kamera tidak ditemukan di perangkat ini.'
      : n === 'NotReadableError' ? 'Kamera sedang dipakai aplikasi lain. Tutup aplikasi itu, lalu coba lagi.'
      : 'Kamera belum bisa diaktifkan. Coba lagi.';
    console.error(e);
  }
  renderCamPh();
}
function stopCamera() {
  cancelAnimationFrame(rafId); rafId = 0;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  el.video.srcObject = null;
  state.cam.active = false; targetBox = drawBox = null;
  el.overlay.getContext('2d').clearRect(0, 0, el.overlay.width, el.overlay.height);
  renderCamPh();
}
el.camBtn.addEventListener('click', enableCamera);

let prevT = 0;
function loop(now) {
  if (!state.cam.active) return;
  const dt = prevT ? now - prevT : 16; prevT = now;
  if (DEMO) { drawDemoScene(now); demoDetect(now); }
  else if (el.video.readyState >= 2 && el.video.currentTime !== lastVT) {
    lastVT = el.video.currentTime;
    const res = detector.detectForVideo(el.video, now);
    metrics.det++;
    const d = res.detections && res.detections[0];
    if (d && d.boundingBox) { const b = d.boundingBox; targetBox = [b.originX, b.originY, b.width, b.height]; faceAt = now; if (!drawBox) drawBox = targetBox.slice(); }
  }
  drawOverlay(dt, now);
  rafId = requestAnimationFrame(loop);
}
function drawOverlay(dt, now) {
  const ctx = el.overlay.getContext('2d');
  ctx.clearRect(0, 0, el.overlay.width, el.overlay.height);
  if (targetBox && now - faceAt < CONFIG.faceStaleMs) {
    const a = 1 - Math.exp(-dt / CONFIG.smoothTauMs);
    for (let i = 0; i < 4; i++) drawBox[i] += (targetBox[i] - drawBox[i]) * a;
    ctx.strokeStyle = accent; ctx.lineWidth = Math.max(3, el.overlay.width / 260);
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(drawBox[0], drawBox[1], drawBox[2], drawBox[3], 14); else ctx.rect(...drawBox);
    ctx.stroke();
  } else { targetBox = null; drawBox = null; }
}
function startLoops() { prevT = 0; cancelAnimationFrame(rafId); rafId = requestAnimationFrame(loop); }

/* ---------------------------------------------------------------- mode demo (tanpa kamera) */
const demo = { p: [.2, .4, .25, .15], dropUntil: 0, nextDrop: 6000 };
function drawDemoScene(now) {
  const c = el.demoCanvas, g = c.getContext('2d'), W = c.width, H = c.height;
  const grad = g.createLinearGradient(0, 0, W, H); grad.addColorStop(0, '#2c4a48'); grad.addColorStop(1, '#16292a');
  g.fillStyle = grad; g.fillRect(0, 0, W, H);
  const sway = Math.sin(now / 1800) * 14;
  g.fillStyle = '#4d6f6b'; g.beginPath(); g.ellipse(W / 2 + sway, H + 20, 150, 110, 0, Math.PI, 0); g.fill();
  g.fillStyle = '#7fa39e'; g.beginPath(); g.ellipse(W / 2 + sway, H / 2 - 10, 62, 78, 0, 0, Math.PI * 2); g.fill();
  g.fillStyle = 'rgba(255,255,255,.7)'; g.font = '600 15px system-ui, sans-serif'; g.fillText('Demo: bukan kamera sungguhan', 14, 24);
}
function demoDetect(now) {
  if (now < demo.dropUntil) return;
  if (now > demo.nextDrop) { demo.dropUntil = now + 1400; demo.nextDrop = now + 14000 + Math.random() * 14000; return; }
  const sway = Math.sin(now / 1800) * 14;
  targetBox = [640 / 2 + sway - 74, 360 / 2 - 10 - 96, 148, 172]; faceAt = now; if (!drawBox) drawBox = targetBox.slice();
}
const DEMO_TARGETS = [[.08, .5, .27, .15], [.3, .28, .12, .3], [.07, .28, .45, .2], [.15, .35, .3, .2], [.28, .3, .1, .32]];
function demoProbs(q) {
  const t = DEMO_TARGETS[q % DEMO_TARGETS.length];
  demo.p = demo.p.map((v, i) => Math.max(0.01, v + (t[i] - v) * 0.1 + (Math.random() - .5) * 0.07));
  const s = demo.p.reduce((a, b) => a + b, 0); demo.p = demo.p.map(v => v / s);
  return demo.p.slice();
}

/* ---------------------------------------------------------------- klasifikasi (kirim crop wajah) */
const cropCv = document.createElement('canvas'); cropCv.width = cropCv.height = CONFIG.cropSize;
const cropCtx = cropCv.getContext('2d');
const clientId = (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Math.random()).slice(2);
let inFlight = 0, sendSeq = 0, appliedSeq = 0;
const ema = (o, v) => (o ? o * 0.8 + v * 0.2 : v);

async function sendCrop(box, ctx) {
  const [x, y, w, h] = box, V = el.video;
  const mx = w * CONFIG.marginRatio, my = h * CONFIG.marginRatio;
  const sx = Math.max(0, Math.floor(x - mx)), sy = Math.max(0, Math.floor(y - my));
  const sw = Math.min(V.videoWidth, Math.ceil(x + w + mx)) - sx, sh = Math.min(V.videoHeight, Math.ceil(y + h + my)) - sy;
  if (sw < 20 || sh < 20) return;
  cropCtx.drawImage(V, sx, sy, sw, sh, 0, 0, cropCv.width, cropCv.height);
  const blob = await new Promise(r => cropCv.toBlob(r, 'image/jpeg', CONFIG.jpegQuality));
  metrics.kb = blob.size / 1024;
  const ctrl = new AbortController(), timer = setTimeout(() => ctrl.abort(), CONFIG.requestTimeoutMs), t0 = performance.now();
  try {
    const res = await fetch('/classify_face', { method: 'POST', headers: { 'Content-Type': 'image/jpeg', 'X-Client-Id': clientId }, body: blob, signal: ctrl.signal });
    const data = await res.json();
    metrics.rtt = ema(metrics.rtt, performance.now() - t0);
    if (data.ms !== undefined) metrics.server = ema(metrics.server, data.ms);
    if (data.infer_ms !== undefined) metrics.infer = ema(metrics.infer, data.infer_ms);
    if (data.wait_ms !== undefined) metrics.wait = ema(metrics.wait, data.wait_ms);
    if (data.probs) onProbs(CLASSES.map(c => Number(data.probs[c.key])), ctx);
  } finally { clearTimeout(timer); }
}

function classifyTick() {
  const s = state.session;
  if (!s || state.screen !== 'session' || s.paused || !state.cam.active || !faceVisible() || inFlight >= CONFIG.maxInFlight) return;
  inFlight++;
  const ctx = { q: s.q, t: s.activeMs, seq: ++sendSeq };
  const job = DEMO ? Promise.resolve().then(() => onProbs(demoProbs(s.q), ctx)) : sendCrop(targetBox.slice(), ctx);
  job.catch(e => console.error('Klasifikasi gagal:', e)).finally(() => { inFlight--; });
}
setInterval(classifyTick, CONFIG.minIntervalMs);

let latest = { p: null, at: 0 };
function onProbs(p, ctx) {
  const s = state.session;
  if (!s || p.some(v => !Number.isFinite(v))) return;
  s.samples.push({ t: ctx.t, q: ctx.q, p });
  if (ctx.seq > appliedSeq) { appliedSeq = ctx.seq; latest = { p, at: performance.now() }; renderMeter(); }
}

/* ---------------------------------------------------------------- meter langsung */
el.meterRows.innerHTML = CLASSES.map(c => `
  <div class="meter-row"><span>${c.label}</span><span class="track"><span class="fill pat-${c.id}" data-k="${c.id}"></span></span><span class="pct" data-p="${c.id}">0%</span></div>`).join('');
function renderMeter() {
  const fresh = latest.p && performance.now() - latest.at < 1500 && faceVisible();
  el.meter.dataset.idle = String(!fresh);
  const p = fresh ? latest.p : [0, 0, 0, 0];
  CLASSES.forEach((c, i) => {
    $(`[data-k=${c.id}]`, el.meterRows).style.width = (p[i] * 100).toFixed(0) + '%';
    $(`[data-p=${c.id}]`, el.meterRows).textContent = Math.round(p[i] * 100) + '%';
  });
  el.meterNow.textContent = fresh ? CLASSES[argmax(p)].label : 'belum ada';
}

/* ---------------------------------------------------------------- persiapan */
function updateStartBtn() {
  const ready = state.cam.active && el.consent.checked;
  el.startBtn.setAttribute('aria-disabled', String(!ready));
  el.startHint.textContent = ready ? 'Siap. Anda bisa menjeda atau mengakhiri sesi kapan saja.'
    : !state.cam.active ? 'Aktifkan kamera dan centang persetujuan untuk mulai.' : 'Centang persetujuan untuk mulai.';
}
el.consent.addEventListener('change', updateStartBtn);
el.startBtn.addEventListener('click', () => {
  if (el.startBtn.getAttribute('aria-disabled') === 'true') { (state.cam.active ? el.consent : el.camBtn.hidden ? el.startBtn : el.camBtn).focus(); return; }
  beginSession();
});

/* ---------------------------------------------------------------- sesi */
function beginSession() {
  const g = n => $(`input[name=${n}]:checked`).value;
  const n = Number(g('count')), seconds = Number(g('seconds'));
  state.previewHidden = g('preview') === 'hide';
  latest = { p: null, at: 0 }; appliedSeq = 0;
  state.session = {
    startedAt: new Date(), questions: QUESTIONS.slice(0, n), limitMs: seconds * 1000, q: 0, activeMs: 0,
    qElapsed: Array(n).fill(0), qStart: Array(n).fill(0), qActive: Array(n).fill(0), qFace: Array(n).fill(0),
    timedOut: Array(n).fill(false), warned: Array(n).fill(false), paused: false, samples: [], demo: DEMO
  };
  go('session');
  showQuestion(0);
}
function showQuestion(i) {
  const s = state.session, n = s.questions.length;
  s.q = i; s.qStart[i] = s.activeMs;
  el.progressText.textContent = `Pertanyaan ${i + 1} dari ${n}`;
  el.qBar.setAttribute('aria-valuemax', n); el.qBar.setAttribute('aria-valuenow', i + 1);
  el.qBarFill.style.width = ((i + 1) / n * 100) + '%';
  el.questionText.textContent = s.questions[i].text;
  el.questionTip.textContent = 'Saran: ' + s.questions[i].tip;
  el.timeNotice.hidden = true; el.timeNotice.textContent = '';
  el.nextBtn.textContent = i === n - 1 ? 'Selesai dan lihat ringkasan' : 'Pertanyaan berikutnya';
  renderTimer();
  el.questionText.focus({ preventScroll: true });
}
function renderTimer() {
  const s = state.session; if (!s) return;
  if (s.limitMs) { el.timerLabel.textContent = 'Sisa waktu'; el.timerValue.textContent = fmtClock(s.limitMs - s.qElapsed[s.q]); }
  else { el.timerLabel.textContent = 'Berjalan'; el.timerValue.textContent = fmtClock(s.qElapsed[s.q]); }
}
function setPaused(p) {
  const s = state.session; if (!s || s.paused === p) return;
  s.paused = p;
  el.pauseLabel.textContent = p ? 'Lanjutkan' : 'Jeda';
  el.pauseBtn.firstElementChild.innerHTML = p ? '<path d="M7 5l12 7-12 7z"/>' : '<rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/>';
  renderCamPh(); announce(p ? 'Sesi dijeda.' : 'Sesi dilanjutkan.');
}
el.pauseBtn.addEventListener('click', () => setPaused(!state.session.paused));
document.addEventListener('visibilitychange', () => {
  if (document.hidden && state.screen === 'session' && state.session && !state.session.paused) setPaused(true);
});
el.nextBtn.addEventListener('click', () => {
  const s = state.session;
  if (s.q === s.questions.length - 1) finishSession(); else showQuestion(s.q + 1);
});
el.endBtn.addEventListener('click', () => { setPaused(true); el.endDlg.showModal(); });
el.endDlg.addEventListener('close', () => {
  if (el.endDlg.returnValue === 'end') finishSession(); else setPaused(false);
  el.endDlg.returnValue = '';
});
el.previewBtn.addEventListener('click', () => { state.previewHidden = !state.previewHidden; renderCamPh(); });

let lastTick = performance.now(), noFaceSince = 0, tickN = 0;
setInterval(() => {                                    // jam sesi dan pemeliharaan UI
  const now = performance.now(), dt = Math.min(now - lastTick, 1000); lastTick = now; tickN++;
  const s = state.session;
  if (s && state.screen === 'session' && !s.paused) {
    s.activeMs += dt; s.qElapsed[s.q] += dt; s.qActive[s.q] += dt;
    if (faceVisible()) s.qFace[s.q] += dt;
    renderTimer();
    if (s.limitMs && !s.timedOut[s.q] && s.qElapsed[s.q] >= s.limitMs) {
      s.timedOut[s.q] = true; el.timeNotice.hidden = false;
      el.timeNotice.textContent = 'Waktu menjawab habis. Lanjutkan saat Anda siap.';
    } else if (s.limitMs && !s.warned[s.q] && s.limitMs > 40000 && s.limitMs - s.qElapsed[s.q] <= 30000) { s.warned[s.q] = true; announce('Sisa 30 detik.'); }
  }
  // petunjuk wajah
  const vis = state.cam.active && faceVisible();
  $('#chkFace').dataset.ok = String(!!vis);
  $('#chkFaceText').textContent = vis ? 'Wajah terlihat' : 'Wajah belum terlihat';
  if (state.cam.active && !vis) { noFaceSince = noFaceSince || now; } else noFaceSince = 0;
  if (state.cam.active && noFaceSince && now - noFaceSince > 1500 && !(s && s.paused))
    el.camMsg.textContent = 'Wajah belum terlihat. Hadap kamera dan pastikan cahaya cukup.';
  else if (state.cam.active) el.camMsg.textContent = '';
  if (state.screen === 'session') renderMeter();
  if (tickN % 4 === 0) {
    accent = getComputedStyle(el.root).getPropertyValue('--accent').trim() || accent;
    metrics.detFps = metrics.det; metrics.det = 0;
    el.techOut.textContent = state.cam.active
      ? `Deteksi lokal ${metrics.detFps} FPS, ping ${metrics.ping.toFixed(0)} ms, RTT klasifikasi ${metrics.rtt.toFixed(0)} ms, server ${metrics.server.toFixed(0)} ms (inferensi ${metrics.infer.toFixed(0)}, antre ${metrics.wait.toFixed(0)}), kirim ${metrics.kb.toFixed(1)} KB.`
      : 'Belum ada data.';
  }
}, 250);
setInterval(async () => {                              // ping kosong: memisahkan biaya jaringan dari inferensi
  if (!state.cam.active || DEMO) return;
  const t0 = performance.now();
  try { await fetch('/ping', { cache: 'no-store' }); metrics.ping = performance.now() - t0; } catch { /* abaikan */ }
}, 2000);

/* ---------------------------------------------------------------- ringkasan: perhitungan */
function buildReport(s) {
  const nQ = Math.min(s.q + 1, s.questions.length);
  const samples = [...s.samples].sort((a, b) => a.t - b.t);
  const overall = Array(K).fill(0), perQ = Array.from({ length: nQ }, () => Array(K).fill(0));
  const nBins = Math.max(1, Math.ceil(s.activeMs / CONFIG.binMs)), bins = Array.from({ length: nBins }, () => Array(K).fill(0));
  samples.forEach((m, i) => {
    const next = samples[i + 1], w = Math.max(1, next ? Math.min(next.t - m.t, 1000) : 400), d = argmax(m.p);
    overall[d] += w; if (perQ[m.q]) perQ[m.q][d] += w; bins[Math.min(nBins - 1, Math.floor(m.t / CONFIG.binMs))][d] += w;
  });
  const total = overall.reduce((a, b) => a + b, 0);
  const activeAll = s.qActive.slice(0, nQ).reduce((a, b) => a + b, 0), faceAll = s.qFace.slice(0, nQ).reduce((a, b) => a + b, 0);
  const questions = perQ.map((v, i) => {
    const sum = v.reduce((a, b) => a + b, 0), pc = roundShares(v);
    return { index: i, text: s.questions[i].text, activeMs: s.qActive[i], coverage: s.qActive[i] ? s.qFace[i] / s.qActive[i] : 0,
      shares: pc, hasData: sum > 0, dominant: sum > 0 ? argmax(v) : null, tense: sum > 0 ? v[2] / sum : 0 };
  });
  const r = {
    startedAt: s.startedAt, demo: s.demo, activeMs: s.activeMs, nQ, nQuestions: s.questions.length, sampleCount: samples.length,
    coverage: activeAll ? faceAll / activeAll : 0, hasData: total > 0,
    shares: roundShares(overall), dominant: total > 0 ? argmax(overall) : null,
    questions, bins: bins.map(b => (b.reduce((a, c) => a + c, 0) > 0 ? argmax(b) : null)),
    qStart: s.qStart.slice(0, nQ), limitMs: s.limitMs
  };
  r.notes = buildNotes(r);
  return r;
}

function buildNotes(r) {
  const out = [], L = i => CLASSES[i].label.toLowerCase();
  if (!r.hasData) return out;
  const tense = r.shares[2], happy = r.shares[0], neutral = r.shares[1];
  if (tense >= 35) out.push({ t: 'Ketegangan cukup sering terbaca', d: `Ekspresi tegang terbaca ${tense}% dari waktu. Sebelum menjawab, tarik napas perlahan dan turunkan bahu. Itu sering membantu wajah ikut rileks.` });
  else if (tense <= 15) out.push({ t: 'Ketegangan jarang terbaca', d: `Ekspresi tegang hanya terbaca ${tense}% dari waktu. Pertahankan ritme dan cara bernapas yang Anda pakai.` });
  else out.push({ t: 'Ketegangan muncul sesekali', d: `Ekspresi tegang terbaca ${tense}% dari waktu. Itu wajar saat berpikir atau mengingat sesuatu.` });
  const withData = r.questions.filter(q => q.hasData);
  if (withData.length >= 2) {
    const top = withData.reduce((a, b) => (b.tense > a.tense ? b : a)), avg = withData.reduce((a, q) => a + q.tense, 0) / withData.length;
    if (top.tense - avg >= 0.10 && top.tense >= 0.25)
      out.push({ t: `Pertanyaan ${top.index + 1} paling menantang`, d: `Ketegangan terbaca ${Math.round(top.tense * 100)}% saat menjawab "${top.text.replace(/[.?!]+$/, '')}". Coba jawab pertanyaan ini sekali lagi, dimulai dari kalimat pembuka yang sudah Anda siapkan.` });
  }
  if (withData.length >= 2) {
    const half = Math.ceil(withData.length / 2), mean = a => a.reduce((x, q) => x + q.tense, 0) / a.length;
    const d = mean(withData.slice(half)) - mean(withData.slice(0, half));
    if (d <= -0.12) out.push({ t: 'Anda makin santai', d: 'Ketegangan turun di paruh kedua sesi. Sesi latihan berikutnya bisa dimulai dengan satu pertanyaan pemanasan.' });
    else if (d >= 0.12) out.push({ t: 'Ketegangan naik di paruh kedua', d: 'Ini bisa tanda lelah atau pertanyaan yang lebih sulit. Jeda singkat di antara pertanyaan bisa membantu.' });
  }
  if (neutral >= 60) out.push({ t: 'Wajah netral mendominasi', d: 'Wajah netral wajar saat Anda fokus berpikir atau menyusun jawaban. Jika ingin terlihat lebih hangat, sebuah senyum tipis di awal atau akhir jawaban cukup, dan itu pilihan Anda.' });
  else if (happy < 8 && tense < 35) out.push({ t: 'Senyum jarang terbaca', d: 'Itu tidak apa-apa. Setiap orang punya gaya sendiri, dan tidak ada ekspresi yang wajib muncul.' });
  if (r.coverage < 0.6) out.unshift({ t: 'Ringkasan ini kurang lengkap', d: `Wajah hanya terdeteksi ${Math.round(r.coverage * 100)}% dari waktu menjawab. Coba cahaya dari depan atau samping, dan posisikan wajah di tengah kamera.` });
  return out;
}

/* ---------------------------------------------------------------- ringkasan: tampilan */
const seg = (shares, extra = '') => `<div class="stack" ${extra} role="img" aria-label="${esc(shares.map((v, i) => `${CLASSES[i].label} ${v}%`).join(', '))}">${
  shares.map((v, i) => `<span class="pat-${CLASSES[i].id}" style="flex:${v} 1 0"></span>`).join('')}</div>`;

function renderReport(r) {
  const d = r.startedAt, dateStr = d.toLocaleDateString('id-ID', { day: 'numeric', month: 'long', year: 'numeric' }),
    timeStr = d.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' }).replace('.', ':');
  el.reportMeta.textContent = `${r.demo ? 'Data simulasi (mode demo). ' : ''}Sesi ${dateStr}, pukul ${timeStr}. Waktu menjawab ${fmtLong(r.activeMs)} untuk ${r.nQ} dari ${r.nQuestions} pertanyaan.`;
  if (!r.hasData) {
    el.reportBody.innerHTML = `<section class="empty"><h2>Belum ada data ekspresi</h2><p class="sub">Wajah belum terbaca selama sesi ini. Pastikan wajah terlihat di kamera dengan cahaya cukup, lalu coba lagi.</p></section>`;
    return;
  }
  const order = r.shares.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
  const head = `Paling sering terbaca ${CLASSES[order[0][1]].label.toLowerCase()} (${order[0][0]}% dari waktu)` +
    (order[1][0] >= 20 ? `, lalu ${CLASSES[order[1][1]].label.toLowerCase()} (${order[1][0]}%).` : '.');
  const legend = CLASSES.map((c, i) => `<li><span class="sw pat-${c.id}"></span><span class="nm">${c.label}<span class="ds">${c.desc}</span></span><span class="vl">${r.shares[i]}%</span></li>`).join('');
  const qs = r.questions.map(q => `<li>
      <p class="qt">${q.index + 1}. ${esc(q.text)}</p>
      ${q.hasData ? seg(q.shares) : '<p class="qm">Wajah tidak terbaca pada pertanyaan ini.</p>'}
      <p class="qm">${q.hasData ? `Paling sering ${CLASSES[q.dominant].label.toLowerCase()}, tegang ${q.shares[2]}%. ` : ''}Waktu menjawab ${fmtLong(q.activeMs)}.</p></li>`).join('');
  const ticks = r.qStart.map((t, i) => `<b style="left:${(t / Math.max(1, r.activeMs) * 100).toFixed(2)}%">P${i + 1}</b>`).join('');
  const ribbon = r.bins.map(b => `<i class="${b === null ? 'pat-none' : 'pat-' + CLASSES[b].id}"></i>`).join('');
  const ribbonLabel = `Garis waktu sesi ${fmtLong(r.activeMs)}. Warna dan pola menunjukkan ekspresi yang paling sering terbaca tiap ${CONFIG.binMs / 1000} detik. Rinciannya ada di tabel di bawah.`;
  const notes = r.notes.map(n => `<li><b>${esc(n.t)}</b>${esc(n.d)}</li>`).join('');
  const rows = r.questions.map(q => `<tr><td>${q.index + 1}. ${esc(q.text)}</td><td class="num">${fmtClock(q.activeMs)}</td>${q.shares.map(v => `<td class="num">${q.hasData ? v + '%' : '-'}</td>`).join('')}<td class="num">${Math.round(q.coverage * 100)}%</td></tr>`).join('');
  const covPct = Math.round(r.coverage * 100);

  el.reportBody.innerHTML = `
  <section aria-labelledby="r-overview">
    <h2 id="r-overview">Gambaran umum</h2>
    <p class="headline">${head}</p>
    <div style="margin-top:1.4rem">${seg(r.shares)}</div>
    <ul class="legend">${legend}</ul>
  </section>
  <section aria-labelledby="r-q">
    <h2 id="r-q">Per pertanyaan</h2>
    <ul class="qlist">${qs}</ul>
  </section>
  <section aria-labelledby="r-time">
    <h2 id="r-time">Sepanjang sesi</h2>
    <div class="ribbon-wrap">
      <div class="ribbon" role="img" aria-label="${esc(ribbonLabel)}">${ribbon}</div>
      <div class="ribbon-ticks" aria-hidden="true">${ticks}</div>
      <div class="ribbon-axis" aria-hidden="true"><span>0:00</span><span>${fmtClock(r.activeMs)}</span></div>
    </div>
    <p class="sub">Garis tegak menandai awal tiap pertanyaan (P1, P2, dan seterusnya). Bagian bergaris tipis berarti wajah tidak terbaca.</p>
    <details class="data"><summary>Lihat data dalam tabel</summary>
      <table><thead><tr><th scope="col">Pertanyaan</th><th class="num" scope="col">Waktu</th>${CLASSES.map(c => `<th class="num" scope="col">${c.label}</th>`).join('')}<th class="num" scope="col">Wajah terbaca</th></tr></thead><tbody>${rows}</tbody></table>
    </details>
  </section>
  ${notes ? `<section aria-labelledby="r-notes"><h2 id="r-notes">Catatan untuk Anda</h2><ul class="notes">${notes}</ul></section>` : ''}
  <section aria-labelledby="r-rel">
    <h2 id="r-rel">Seberapa bisa diandalkan</h2>
    <div class="reliability">
      <p><strong>Wajah terbaca ${covPct}% dari waktu menjawab</strong> (${r.sampleCount} pembacaan).</p>
      <p>Ini perkiraan dari model yang dilatih pada foto wajah. Ketepatannya pada sistem ini belum diuji.</p>
      <p>Cahaya redup, kacamata, masker, dan sudut kamera memengaruhi hasil. Cara orang berekspresi juga berbeda karena budaya, kebiasaan, atau kondisi yang memengaruhi gerak wajah, dan model tidak memperhitungkannya.</p>
    </div>
  </section>`;
}

function finishSession() {
  const s = state.session; if (!s) return;
  s.endedAt = new Date();
  stopCamera();
  state.report = buildReport(s);
  renderReport(state.report);
  go('report');
}

/* ---------------------------------------------------------------- aksi ringkasan */
$('#againBtn').addEventListener('click', () => { state.session = null; state.report = null; go('setup'); });
$('#printBtn').addEventListener('click', () => window.print());
$('#wipeBtn').addEventListener('click', () => {
  state.session = null; state.report = null; el.reportBody.innerHTML = ''; el.reportMeta.textContent = '';
  go('setup'); announce('Data sesi sudah dihapus dari peramban ini.');
});
$('#jsonBtn').addEventListener('click', () => {
  const s = state.session, r = state.report; if (!s || !r) return;
  const data = {
    versi: 1, dibuat: s.startedAt.toISOString(), demo: s.demo,
    pengaturan: { jumlahPertanyaan: s.questions.length, batasWaktuDetik: s.limitMs / 1000 },
    kelas: CLASSES.map(c => c.key),
    ringkasan: { persen: Object.fromEntries(CLASSES.map((c, i) => [c.key, r.shares[i]])), wajahTerbaca: +r.coverage.toFixed(3), waktuMenjawabMs: Math.round(r.activeMs) },
    perPertanyaan: r.questions.map(q => ({ no: q.index + 1, teks: q.text, waktuMs: Math.round(q.activeMs), persen: Object.fromEntries(CLASSES.map((c, i) => [c.key, q.shares[i]])) })),
    sampel: s.samples.map(m => ({ t: Math.round(m.t), q: m.q + 1, p: m.p.map(v => +v.toFixed(4)) }))
  };
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  a.download = `ringkasan-sesi-${s.startedAt.toISOString().slice(0, 16).replace(/[:T]/g, '-')}.json`;
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

/* ---------------------------------------------------------------- mulai */
loadPrefs();
if (DEMO) { $('#demoBanner').hidden = false; }
go('setup', false);
