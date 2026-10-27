#!/usr/bin/python

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))  # Folder deployment
root_dir = os.path.dirname(current_dir)                  # Folder AffectNet

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import time
from collections import deque
import cv2
import keras.backend as K
import numpy as np
import tensorflow as tf

from deployment.tensorflow_detector import *
from deployment.utils import label_map_util
from deployment.utils import visualization_utils_color as vis_util
from deployment.video_threading_optimization import *

# IMPORT WRAPPER KELOMPOK
from interview_model_wrapper import InterviewEmotionPredictor

## OPTIONS ##
PATH_TO_CKPT = './frozen_graphs/frozen_inference_graph_face.pb'
PATH_TO_CLASS = './frozen_graphs/classificator_full_model.pb'
PATH_TO_REGRESS = './frozen_graphs/regressor_full_model.pb'
label_map = label_map_util.load_labelmap('./protos/face_label_map.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=2, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Inisialisasi Predictor & Buffer Smoothing (5 frame terakhir)
predictor = InterviewEmotionPredictor()
history_length = 5
prob_buffer = deque(maxlen=history_length)


def predict_from_camera(detector):
    print('Press q to exit')
    vs = WebcamVideoStream(src=0).start()
    fps = FPS().start()
    window_not_set = True

    while True:
        image = vs.read()
        if image is None:
            continue
            
        [h, w] = image.shape[:2]
        image = cv2.flip(image, 1)

        # Jalankan detektor bawaan
        boxes, scores, classes, num_detections, raw_emotions = detector.run(image)

        # Pemetaan 8 Emosi -> 4 Emosi Wawancara
        if raw_emotions is not None and len(raw_emotions) > 0:
            _, _, norm_probs = predictor.map_emotions(raw_emotions)
            prob_buffer.append(norm_probs)
            
            # Temporal Smoothing (Moving Average)
            smoothed_probs = np.mean(prob_buffer, axis=0)
            final_idx = np.argmax(smoothed_probs)
            final_label = predictor.target_classes[final_idx]
            confidence = smoothed_probs[final_idx] * 100
            
            display_text = f"Emosi: {final_label} ({confidence:.1f}%)"
        else:
            display_text = "Mendeteksi wajah..."

        # Tampilkan teks emosi target di layar
        cv2.putText(image, display_text, org=(30, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8, color=(0, 255, 0), thickness=2)

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=1)

        if window_not_set is True:
            cv2.namedWindow("Simulasi Wawancara - Affective Computing", cv2.WINDOW_NORMAL)
            window_not_set = False

        cv2.imshow("Simulasi Wawancara - Affective Computing", image)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break
        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()
    vs.stop()


def predict_from_video(detector, file_path):
    vs = FileVideoStream(file_path).start()
    time.sleep(2.0)
    fps = FPS().start()
    out = None
    frame_count = 0

    while vs.more():
        image = vs.read()
        frame_count += 1

        if out is None:
            [h, w] = image.shape[:2]
            out = cv2.VideoWriter("test_out.avi", cv2.VideoWriter_fourcc(*'MJPG'), 25.0, (w, h))

        if frame_count % 2 == 0:
            boxes, scores, classes, num_detections, raw_emotions = detector.run(image)

            if raw_emotions is not None and len(raw_emotions) > 0:
                _, _, norm_probs = predictor.map_emotions(raw_emotions)
                prob_buffer.append(norm_probs)
                
                smoothed_probs = np.mean(prob_buffer, axis=0)
                final_idx = np.argmax(smoothed_probs)
                final_label = predictor.target_classes[final_idx]
                confidence = smoothed_probs[final_idx] * 100
                
                display_text = f"Emosi: {final_label} ({confidence:.1f}%)"
            else:
                display_text = "Mendeteksi..."

            cv2.putText(image, display_text, org=(30, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8, color=(0, 255, 0), thickness=2)

            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=1)

        out.write(image)
        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '-c':
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        detector = TensorflowDetector(PATH_TO_CKPT, PATH_TO_CLASS, PATH_TO_REGRESS)
        predict_from_camera(detector)

    elif len(sys.argv) > 1 and sys.argv[1] == '-v':
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        detector = TensorflowDetector(PATH_TO_CKPT, PATH_TO_CLASS, PATH_TO_REGRESS)
        predict_from_video(detector, 'test.mp4')

    else:
        print('Gunakan argumen -c untuk webcam atau -v untuk video mp4')