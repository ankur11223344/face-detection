import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_nose.xml')
webcam = cv2.VideoCapture(0)

while True:

    successful_frame_read, frame = webcam.read()
    if not successful_frame_read:
        break

    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame_grayscale)



    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)
        the_face = frame[y:y+h, x:x+w]
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        smiles = smile_cascade.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)

        eyes = eye_cascade.detectMultiScale(face_grayscale, scaleFactor=1.1, minNeighbors=15)

        nose = nose_cascade.detectMultiScale(face_grayscale, scaleFactor=1.1, minNeighbors=15)

        for(bx, by, bw, bh) in smiles:
            cv2.rectangle(the_face, (bx, by), ((bx+bw), (by+bh)), (255, 50, 50), 2)

        for (ax, ay, aw, ah) in nose:
            cv2.rectangle(the_face, (ax, ay), (ax+aw, ay+ah),(255, 255, 255), 2)

            for(x_, y_, w_, h_) in eyes:
                cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (255, 255, 255), 3)

        if len(nose) > 0:
            cv2.putText(frame, 'nose_detected', (y+h, x+w+20), fontFace=2, fontScale=cv2.FONT_HERSHEY_PLAIN,
                        color=(255,255,255))

        if len(eyes) > 0:
            cv2.putText(frame, 'eyes_detected', (y + h + 80, x), fontFace=2, fontScale=cv2.FONT_HERSHEY_PLAIN,
                        color=(255, 255, 255))

        if len(smiles) > 0:
            cv2.putText(frame, 'smile_detected', (x, y+h+40), fontScale=2, fontFace=cv2.FONT_HERSHEY_PLAIN,
                        color=(255, 255, 255))



    cv2.imshow('Smile Detector', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()


