import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
from flask import Flask, jsonify, request

root_dir = os.path.dirname(os.path.abspath(__file__))
deployment_dir = os.path.join(root_dir, 'deployment')
for p in (root_dir, deployment_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

_t_start = time.time()
print('[startup] mengimpor TensorFlow/Keras...', flush=True)
from emotion_classifier import EmotionClassifier, EMOTION_CLASSES
print(f'[startup] impor selesai: {time.time() - _t_start:.1f} s', flush=True)
from interview_model_wrapper import InterviewEmotionPredictor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024   # crop wajah kecil; tolak payload besar

PATH_TO_CLASS = os.path.join(deployment_dir, 'frozen_graphs', 'classificator_full_model.pb')

BUFFER_SIZE = 5
BUFFER_RESET_SECONDS = 1.5   # jika jeda antar crop lebih lama (wajah hilang), buffer dikosongkan
CLIENT_TTL_SECONDS = 300

_t_model = time.time()
classifier = EmotionClassifier(PATH_TO_CLASS)
print(f'[startup] model dimuat + pemanasan: {time.time() - _t_model:.1f} s', flush=True)
predictor = InterviewEmotionPredictor()
infer_lock = threading.Lock()


class ClientState:
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.last_pred = 0.0
        self.last_seen = time.time()


clients = {}
clients_lock = threading.Lock()


def get_state(client_id):
    now = time.time()
    with clients_lock:
        for k in [k for k, v in clients.items() if now - v.last_seen > CLIENT_TTL_SECONDS]:
            del clients[k]
        st = clients.get(client_id)
        if st is None:
            st = ClientState()
            clients[client_id] = st
        st.last_seen = now
        return st


HTML_PAGE = """<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Simulasi Wawancara Real-Time</title>
<style>
  body { font-family: Arial, sans-serif; text-align: center; background: #121212; color: #fff; margin: 0; padding: 20px; }
  .video-container { position: relative; width: 100%; max-width: 640px; margin: 0 auto;
                     border-radius: 8px; overflow: hidden; border: 2px solid #2ecc71; }
  video { width: 100%; height: auto; display: block; transform: scaleX(-1); }
  canvas#overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%;
                   pointer-events: none; transform: scaleX(-1); }
  .status { margin-top: 15px; font-weight: bold; font-size: 1.2em; color: #2ecc71; }
  .debug { margin-top: 6px; font-size: 0.8em; color: #888; }
</style>
</head>
<body>
<h2>Simulasi Wawancara | MobileNetV2 Emotion AI</h2>
<div class="video-container">
  <video id="video" autoplay playsinline muted></video>
  <canvas id="overlay"></canvas>
</div>
<div class="status" id="status">Memuat model deteksi wajah...</div>
<div class="debug" id="debug"></div>

<script type="module">
// Ubah ke true jika berkas MediaPipe sudah disalin ke folder static/mediapipe/ (tanpa CDN pihak ketiga).
const SELF_HOST = false;
const MP = SELF_HOST ? {
  bundle: '/static/mediapipe/vision_bundle.mjs',
  wasm:   '/static/mediapipe/wasm',
  model:  '/static/mediapipe/blaze_face_short_range.tflite'
} : {
  bundle: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs',
  wasm:   'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm',
  model:  'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'
};

const CONFIG = {
  cropSize: 224, marginRatio: 0.10, jpegQuality: 0.9,
  minIntervalMs: 150,        // jeda minimum antar kiriman crop ke server
  maxInFlight: 2,            // jumlah request klasifikasi yang boleh berjalan bersamaan
  requestTimeoutMs: 2500,
  faceStaleMs: 500,          // wajah dianggap hilang jika tidak terdeteksi selama ini
  smoothTauMs: 40,           // interpolasi kotak (ringan, deteksi sudah berjalan per frame)
  minFaceScore: 0.5
};

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const statusText = document.getElementById('status');
const debugText = document.getElementById('debug');
const octx = overlay.getContext('2d');

const crop = document.createElement('canvas');   // di luar DOM
crop.width = crop.height = CONFIG.cropSize;
const cropCtx = crop.getContext('2d');

const clientId = (window.crypto && crypto.randomUUID) ? crypto.randomUUID()
                 : String(Math.random()).slice(2);

let detector = null;
let targetBox = null, drawBox = null, lastFaceAt = 0, lastVideoTime = -1;
let detCount = 0, detFps = 0, rttEma = 0, serverEma = 0, inferEma = 0, waitEma = 0, lastKb = 0, pingMs = 0;
setInterval(() => { detFps = detCount; detCount = 0; renderDebug(); }, 1000);

function renderDebug() {
  debugText.innerText =
    `Deteksi lokal ${detFps} FPS | ping ${pingMs.toFixed(0)} ms | klasifikasi RTT ${rttEma.toFixed(0)} ms | server ${serverEma.toFixed(0)} ms (infer ${inferEma.toFixed(0)}, antre ${waitEma.toFixed(0)}) | kirim ${lastKb.toFixed(1)} KB`;
}

async function createDetector(mp, vision) {
  for (const delegate of ['GPU', 'CPU']) {
    try {
      return await mp.FaceDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MP.model, delegate },
        runningMode: 'VIDEO',
        minDetectionConfidence: CONFIG.minFaceScore
      });
    } catch (e) { console.warn('Delegate gagal:', delegate, e); }
  }
  throw new Error('FaceDetector gagal dibuat');
}

async function init() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    statusText.innerText = "Kamera memerlukan HTTPS. Buka lewat alamat https://...";
    return;
  }
  try {
    const mp = await import(MP.bundle);
    const vision = await mp.FilesetResolver.forVisionTasks(MP.wasm);
    detector = await createDetector(mp, vision);
  } catch (e) {
    console.error(e);
    statusText.innerText = "Gagal memuat model deteksi wajah (periksa koneksi / path MediaPipe).";
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false
    });
    video.srcObject = stream;
    await new Promise(res => video.addEventListener('loadedmetadata', res, { once: true }));
    await video.play();
  } catch (e) {
    console.error(e);
    statusText.innerText = "Akses kamera ditolak / tidak didukung!";
    return;
  }
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  statusText.innerText = "Kamera aktif. Mendeteksi wajah...";
  requestAnimationFrame(renderLoop);
  setInterval(classifyTick, CONFIG.minIntervalMs);
}

// Loop 1: deteksi wajah lokal + gambar overlay, setiap frame video baru.
let prevT = performance.now();
function renderLoop() {
  const now = performance.now();
  const dt = now - prevT; prevT = now;

  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const res = detector.detectForVideo(video, now);
    detCount++;
    const d = res.detections && res.detections[0];
    if (d && d.boundingBox) {
      const b = d.boundingBox;
      targetBox = [b.originX, b.originY, b.width, b.height];   // piksel frame video
      lastFaceAt = now;
      if (!drawBox) drawBox = targetBox.slice();
    }
  }

  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (targetBox && now - lastFaceAt < CONFIG.faceStaleMs) {
    const alpha = 1 - Math.exp(-dt / CONFIG.smoothTauMs);
    for (let i = 0; i < 4; i++) drawBox[i] += (targetBox[i] - drawBox[i]) * alpha;
    octx.strokeStyle = '#00FF00';
    octx.lineWidth = 3;
    octx.strokeRect(drawBox[0], drawBox[1], drawBox[2], drawBox[3]);
  } else {
    targetBox = null; drawBox = null;
  }
  requestAnimationFrame(renderLoop);
}

function faceVisible() {
  return targetBox && performance.now() - lastFaceAt < CONFIG.faceStaleMs;
}

// Loop 2: kirim HANYA crop wajah ke server untuk klasifikasi emosi.
async function sendCrop(box, seq) {
  const [x, y, w, h] = box;
  const mx = w * CONFIG.marginRatio, my = h * CONFIG.marginRatio;   // margin, mirip +20 px di detektor lama
  const sx = Math.max(0, Math.floor(x - mx)), sy = Math.max(0, Math.floor(y - my));
  const ex = Math.min(video.videoWidth, Math.ceil(x + w + mx));
  const ey = Math.min(video.videoHeight, Math.ceil(y + h + my));
  const sw = ex - sx, sh = ey - sy;
  if (sw < 20 || sh < 20) return;

  cropCtx.drawImage(video, sx, sy, sw, sh, 0, 0, crop.width, crop.height);
  const blob = await new Promise(r => crop.toBlob(r, 'image/jpeg', CONFIG.jpegQuality));
  lastKb = blob.size / 1024;

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), CONFIG.requestTimeoutMs);
  const t0 = performance.now();
  try {
    const res = await fetch('/classify_face', {
      method: 'POST',
      headers: { 'Content-Type': 'image/jpeg', 'X-Client-Id': clientId },
      body: blob,
      signal: ctrl.signal
    });
    const data = await res.json();
    const rtt = performance.now() - t0;
    rttEma = rttEma ? rttEma * 0.8 + rtt * 0.2 : rtt;
    const ema = (old, v) => old ? old * 0.8 + v * 0.2 : v;
    if (data.ms !== undefined) serverEma = ema(serverEma, data.ms);
    if (data.infer_ms !== undefined) inferEma = ema(inferEma, data.infer_ms);
    if (data.wait_ms !== undefined) waitEma = ema(waitEma, data.wait_ms);
    // Abaikan jawaban yang lebih lama dari yang sudah ditampilkan (request bisa selesai tidak berurutan).
    if (data.text && faceVisible() && seq > appliedSeq) { appliedSeq = seq; statusText.innerText = data.text; }
  } finally {
    clearTimeout(timer);
  }
}

let inFlight = 0, sendSeq = 0, appliedSeq = 0;

function classifyTick() {
  if (!faceVisible()) { statusText.innerText = "Mendeteksi wajah..."; return; }
  if (inFlight >= CONFIG.maxInFlight) return;
  inFlight++;
  sendCrop(targetBox.slice(), ++sendSeq)
    .catch(e => console.error('Klasifikasi gagal:', e))
    .finally(() => { inFlight--; });
}

// Ping kosong tiap 2 detik: memisahkan biaya jaringan (koneksi + TLS) dari biaya klasifikasi.
setInterval(async () => {
  const t0 = performance.now();
  try { await fetch('/ping', { cache: 'no-store' }); pingMs = performance.now() - t0; } catch (e) {}
}, 2000);

init();
</script>
</body>
</html>
"""


@app.get('/')
def home():
    return HTML_PAGE


@app.get('/ping')
def ping():
    return ('', 204)


@app.post('/classify_face')
def classify_face():
    t0 = time.perf_counter()
    client_id = request.headers.get('X-Client-Id', 'anon')[:64]

    raw = request.get_data()
    if not raw:
        return jsonify({'error': 'No image provided'}), 400
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape[0] < 20 or bgr.shape[1] < 20:
        return jsonify({'error': 'Invalid face crop'}), 400
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t_decoded = time.perf_counter()

    state = get_state(client_id)
    now = time.time()
    if now - state.last_pred > BUFFER_RESET_SECONDS:
        state.buffer.clear()          # wajah sempat hilang: jangan pakai prediksi lama

    t_wait0 = time.perf_counter()
    with infer_lock:
        t_wait1 = time.perf_counter()
        probs8 = classifier.predict(rgb)   # probabilitas softmax 8 kelas, bukan label string
        t_inf1 = time.perf_counter()

    _, _, norm_probs = predictor.map_emotions(probs8)
    state.buffer.append(norm_probs)
    state.last_pred = now

    smoothed = np.mean(state.buffer, axis=0)
    idx = int(np.argmax(smoothed))
    return jsonify({
        'text': f"Emosi: {predictor.target_classes[idx]} ({smoothed[idx] * 100:.1f}%)",
        'probs': {c: float(p) for c, p in zip(predictor.target_classes, smoothed)},
        'raw_label': EMOTION_CLASSES[int(np.argmax(probs8))],   # label 8 kelas mentah, untuk validasi
        'ms': (time.perf_counter() - t0) * 1000,
        'decode_ms': (t_decoded - t0) * 1000,
        'wait_ms': (t_wait1 - t_wait0) * 1000,    # antre menunggu lock inferensi
        'infer_ms': (t_inf1 - t_wait1) * 1000     # waktu murni klasifikasi
    })


def lan_ip():
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(('10.255.255.255', 1))   # tidak mengirim paket; hanya memilih antarmuka
            return sk.getsockname()[0]
    except Exception:
        return '127.0.0.1'


def ensure_self_signed_cert(cert_path, key_path):
    """Buat sertifikat self-signed (ECDSA P-256, 1 tahun) bila belum ada. Butuh: pip install cryptography"""
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return
    import datetime
    import ipaddress
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'interview-sim-local')])
    san = [x509.DNSName('localhost'), x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]
    try:
        san.append(x509.IPAddress(ipaddress.ip_address(lan_ip())))
    except ValueError:
        pass
    now = datetime.datetime.utcnow()
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .sign(key, hashes.SHA256()))
    os.makedirs(os.path.dirname(cert_path), exist_ok=True)
    with open(key_path, 'wb') as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                  serialization.NoEncryption()))
    with open(cert_path, 'wb') as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))


if __name__ == '__main__':
    use_https = os.environ.get('USE_HTTPS', '1') != '0'
    scheme = 'https' if use_https else 'http'
    print(f'[startup] siap dalam {time.time() - _t_start:.1f} s -> {scheme}://{lan_ip()}:5000', flush=True)

    # Server dev Werkzeug SELALU mengirim "Connection: close": tiap request membuka koneksi baru
    # (TCP + TLS handshake), mahal di Wi-Fi HP. Cheroot mendukung keep-alive dan TLS.
    try:
        from cheroot.wsgi import Server as CherootServer
        have_cheroot = True
    except ImportError:
        have_cheroot = False

    if have_cheroot and os.environ.get('USE_CHEROOT', '1') != '0':
        server = CherootServer(('0.0.0.0', 5000), app, numthreads=8, timeout=30)
        if use_https:
            from cheroot.ssl.builtin import BuiltinSSLAdapter
            cert_path = os.path.join(root_dir, 'certs', 'server.crt')
            key_path = os.path.join(root_dir, 'certs', 'server.key')
            ensure_self_signed_cert(cert_path, key_path)
            server.ssl_adapter = BuiltinSSLAdapter(cert_path, key_path)
        print('[startup] server: cheroot (keep-alive aktif)', flush=True)
        try:
            server.start()
        except KeyboardInterrupt:
            server.stop()
    else:
        print('[startup] server: Flask dev (tanpa keep-alive). Untuk RTT lebih rendah: pip install cheroot', flush=True)
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True,
                ssl_context='adhoc' if use_https else None)
