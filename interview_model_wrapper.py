import numpy as np

class InterviewEmotionPredictor:
    def __init__(self):
        self.target_classes = ['Happy', 'Neutral', 'Tense', 'Calm']
        self.affectnet_labels = {
            'neutral': 0,
            'happy': 1,
            'sad': 2,
            'surprised': 3, 'surprise': 3,
            'afraid': 4, 'fear': 4,
            'disgusted': 5, 'disgust': 5,
            'angry': 6, 'anger': 6,
            'contemptuous': 7, 'contempt': 7
        }

    def map_emotions(self, raw_probs):
        p_vector = np.zeros(8, dtype=np.float32)

        if isinstance(raw_probs, str):
            label_str = raw_probs.strip().lower()
            idx = self.affectnet_labels.get(label_str, 0)
            p_vector[idx] = 1.0

        elif isinstance(raw_probs, (list, tuple, np.ndarray)) and len(raw_probs) > 0:
            first_elem = raw_probs[0]
            if isinstance(first_elem, str):
                label_str = first_elem.strip().lower()
                idx = self.affectnet_labels.get(label_str, 0)
                p_vector[idx] = 1.0
            elif isinstance(first_elem, (list, tuple)) and len(first_elem) > 0 and isinstance(first_elem[0], str):
                label_str = first_elem[0].strip().lower()
                idx = self.affectnet_labels.get(label_str, 0)
                p_vector[idx] = 1.0
            else:
                try:
                    arr = np.array(raw_probs, dtype=np.float32).flatten()
                    if len(arr) >= 8:
                        p_vector = arr[:8]
                    else:
                        p_vector[:len(arr)] = arr
                except ValueError:
                    p_vector[0] = 1.0
        else:
            p_vector[0] = 1.0

        p_neutral = float(p_vector[0])
        p_happy   = float(p_vector[1])
        p_sad     = float(p_vector[2])
        p_surp    = float(p_vector[3])
        p_fear    = float(p_vector[4])
        p_disg    = float(p_vector[5])
        p_anger   = float(p_vector[6])
        p_cont    = float(p_vector[7])

        # --- FORMULASI BERBASIS PERGERAKAN OTOT WAJAH ---
        
        # 1. Neutral: Wajah murni datar tanpa pergerakan otot
        score_neutral = p_neutral * 0.85

        # 2. Calm: Didominasi pergerakan otot mulut (senyum tipis / rileks / Contemptuous halus)
        score_calm = (0.5 * p_neutral) + (0.5 * p_happy if p_happy < 0.6 else 0.2 * p_happy) + (0.3 * p_cont)

        # 3. Happy: Murni kelas Happy saat senyum jelas / terbuka
        score_happy = p_happy * 1.2 if p_happy >= 0.5 else p_happy * 0.4

        # 4. Tense: Otot alis berkontraksi (Corrugator / Procerus) -> Anger, Fear, Sad, Disgust
        eyebrow_movement = np.max([p_anger, p_fear, p_sad, p_disg])
        score_tense = float(eyebrow_movement)

        scores = np.array([score_happy, score_neutral, score_tense, score_calm], dtype=np.float32)

        # Linear Normalization
        total_score = np.sum(scores)
        if total_score > 0:
            norm_probs = scores / total_score
        else:
            norm_probs = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        predicted_idx = int(np.argmax(norm_probs))
        predicted_label = self.target_classes[predicted_idx]
        confidence = float(norm_probs[predicted_idx])

        return predicted_label, confidence, norm_probs
    