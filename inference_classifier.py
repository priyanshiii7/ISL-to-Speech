import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Mediapipe hand detection setup - allow 2 hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Label dictionary - adjust to match your dataset
labels_dict = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g',
    7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm',
    13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's',
    19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'
}

# Text-to-speech setup
engine = pyttsx3.init()

# Variables to store detected letters & words
current_letter = ''
prev_letter = ''
stable_count = 0
letter_stable_threshold = 10  # Frames needed to accept letter

current_word = ''
last_sign_time = time.time()
sign_timeout = 2  # Seconds after last letter to speak word

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

        # Normalize and extract features for all landmarks from both hands
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        if data_aux:
            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_letter = prediction[0]
            except:
                predicted_letter = ''  # In case of feature size mismatch

            if predicted_letter == prev_letter:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count > letter_stable_threshold:
                if predicted_letter != current_letter:
                    current_letter = predicted_letter
                    current_word += current_letter
                    print(f"Current Word: {current_word}")
                    stable_count = 0

                last_sign_time = time.time()

            prev_letter = predicted_letter

            # Show predicted letter
            cv2.putText(frame, f'Letter: {predicted_letter}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, cv2.LINE_AA)

    else:
        # If no hand detected & timeout reached, speak the word
        if current_word and (time.time() - last_sign_time) > sign_timeout:
            print(f"✅ Speaking word: {current_word}")
            engine.say(current_word)
            engine.runAndWait()
            current_word = ''
            current_letter = ''
            prev_letter = ''
            stable_count = 0

    # Show current word
    cv2.putText(frame, f'Word: {current_word}', (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Sign Language to Speech', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
