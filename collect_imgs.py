#this file collects imgs from the webcam and saves to dataset 

import os
import cv2

DATA_DIR = './dataset_ISL'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = ['a', 'b', 'c', 'd' , 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
dataset_size = 100

cap = cv2.VideoCapture(0)

for j in number_of_classes:
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}' .format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready?  Press "Q" !:)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break


    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()