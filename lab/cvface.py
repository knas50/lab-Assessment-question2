import cv2
import sys

import datetime

# fface = './haarcascasde_frontalface_alt2.xml'
#fface = '/home/emma/python-workspace/landmark_tracking/assets/haarcascasde_frontalface_alt2.xml'
#
ff = 'haarcascade_frontalface_alt2.xml'

cap = cv2.VideoCapture(0)

cl = cv2.CascadeClassifier(ff)

if cl.empty():
    print('not loaded')
    sys.exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print('error')
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = cl.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=1, minSize=(60, 60))
    #face_rects = classifier.detectMultiScale(image=gray_frame)


    for rect in face_rects:
        x, y, w, h = rect
        roi = frame[y: y+h, x: x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # save roi
        dt = datetime.datetime.now()
        image = f'face_{dt}.jpg'
        cv2.imwrite(image, roi)

    cv2.imshow('image', frame)

    if cv2.waitKey(1) & 0xff == ord('s'):
        break


cap.release()
cv2.destroyAllWindows()


