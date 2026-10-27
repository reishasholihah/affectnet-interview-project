"""Klasifikator emosi saja (tanpa detektor wajah SSD dan tanpa regressor).

Memuat classificator_full_model.pb dengan cara yang sama seperti TensorflowDetector,
tetapi mengembalikan PROBABILITAS softmax 8 kelas (bukan label string hasil argmax),
sehingga InterviewEmotionPredictor menerima distribusi yang sebenarnya.
"""
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.applications.mobilenet import preprocess_input

# Urutan indeks sama dengan keluaran model dan dengan InterviewEmotionPredictor.affectnet_labels
EMOTION_CLASSES = ['Neutral', 'Happy', 'Sad', 'Surprised', 'Afraid', 'Disgusted', 'Angry', 'Contemptuous']


class EmotionClassifier:
    def __init__(self, path_to_class):
        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(path_to_class, 'rb') as fid:
                graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(graph=self.graph, config=config)
        self.inp = self.graph.get_tensor_by_name('input_1:0')
        self.out = self.graph.get_tensor_by_name('dense_2/Softmax:0')

        # Pemanasan agar request pertama tidak lambat.
        self.predict(np.zeros((224, 224, 3), dtype=np.uint8))

    def predict(self, rgb_face):
        """rgb_face: array uint8 (H, W, 3) berformat RGB berisi crop wajah.
        Mengembalikan vektor probabilitas float32 berukuran 8."""
        img = Image.fromarray(rgb_face).resize((224, 224))   # sama dengan TensorflowDetector.run
        arr = np.asarray(img, dtype=np.float32).copy()
        arr = preprocess_input(arr)
        probs = self.sess.run(self.out, feed_dict={self.inp: arr[np.newaxis, ...]})
        return probs[0]
