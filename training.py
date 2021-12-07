import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor (img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if new_faces is ():
        return None

    for(bx, by, bw, bh) in new_faces:
        cropped_face = img[by:by+bh, bx:bx+bw]

        return cropped_face

webcam =cv2.VideoCapture(0)
Count = 0

while True:
    ret, frame =webcam.read()
    if face_extractor(frame) is not None:
        Count = Count+1
        c_face = cv2.resize(face_extractor(frame), (200, 200))
        c_face = cv2.cvtColor(c_face, cv2.COLOR_BGR2GRAY)

        file_name_path ='C:/Users/Ankur/Desktop/dataset' +str(Count)+'.jpg'
        cv2.imwrite(file_name_path, c_face)

        cv2.putText(c_face, str(Count), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('face cropper', c_face)

    else:
        print('face not found')
        pass

    cv2.waitKey(1)
webcam.release()
cv2.destroyAllWindows()


