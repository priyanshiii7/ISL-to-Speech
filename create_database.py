import mediapipe as mp
import cv2
import os
import pickle

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

DATA_DIR = './dataset_ISL'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            continue  # Skip if image not loaded properly

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # Sort hands to maintain consistent order
            hand_landmarks_list = sorted(results.multi_hand_landmarks, key=lambda hand: hand.landmark[0].x)

            for hand_landmarks in hand_landmarks_list:
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

            for hand_landmarks in hand_landmarks_list:
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

            # If only one hand detected, pad features with zeros for the second hand
            if len(hand_landmarks_list) == 1:
                data_aux.extend([0] * 42)  # 21 landmarks * 2 (x and y) = 42 zeros

            data.append(data_aux)
            labels.append(dir_)

# Save the processed dataset
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("\n✅ Two-hand dataset saved as 'data.pickle'")
