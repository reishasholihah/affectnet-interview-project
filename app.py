import mimetypes
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

# Registri Windows sering memetakan .js/.mjs ke text/plain; wajib diset sebelum Flask/Cheroot
mimetypes.add_type('text/javascript', '.js')
mimetypes.add_type('text/javascript', '.mjs')
mimetypes.add_type('application/wasm', '.wasm')
mimetypes.add_type('text/css', '.css')

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
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0   # Jangan cache berkas statis selama pengembangan
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024   # Crop wajah kecil (~5 KB)

PATH_TO_CLASS = os.path.join(deployment_dir, 'frozen_graphs', 'classificator_full_model.pb')

BUFFER_SIZE = 5
BUFFER_RESET_SECONDS = 1.5   # Jika jeda antar crop lebih lama (wajah hilang), buffer dikosongkan
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


@app.get('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')


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
        state.buffer.clear()

    t_wait0 = time.perf_counter()
    with infer_lock:
        t_wait1 = time.perf_counter()
        probs8 = classifier.predict(rgb)
        t_inf1 = time.perf_counter()

    _, _, norm_probs = predictor.map_emotions(probs8)
    state.buffer.append(norm_probs)
    state.last_pred = now

    smoothed = np.mean(state.buffer, axis=0)
    idx = int(np.argmax(smoothed))
    return jsonify({
        'text': f"Emosi: {predictor.target_classes[idx]} ({smoothed[idx] * 100:.1f}%)",
        'probs': {c: float(p) for c, p in zip(predictor.target_classes, smoothed)},
        'raw_label': EMOTION_CLASSES[int(np.argmax(probs8))],
        'ms': (time.perf_counter() - t0) * 1000,
        'decode_ms': (t_decoded - t0) * 1000,
        'wait_ms': (t_wait1 - t_wait0) * 1000,
        'infer_ms': (t_inf1 - t_wait1) * 1000
    })


def lan_ip():
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(('10.255.255.255', 1))
            return sk.getsockname()[0]
    except Exception:
        return '127.0.0.1'


def ensure_self_signed_cert(cert_path, key_path):
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
        