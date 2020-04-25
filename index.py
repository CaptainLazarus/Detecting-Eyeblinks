import cv2
import imutils
import dlib
import time
import argparse
import numpy as np
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from scipy.spatial import distance as dist
from imutils import face_utils

def initArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v" , "--video" , help="Input video stream")
    parser.add_argument("-p" , "--shape" , required=True)
    return vars(parser.parse_args())

def ear(eye):
    a = dist.euclidean(eye[1] , eye[5])
    b = dist.euclidean(eye[2] , eye[4])
    c = dist.euclidean(eye[0] , eye[3])

    return ((a+b)/(2*c))

EAR_THRESH = 0.2
EAR_FRAMES = 5
COUNTER = 0
TOTAL = 0

if __name__ == "__main__":
    args = initArgs()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args['shape'])

    (lstart , lend) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    (rstart , rend) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

    vs = FileVideoStream(args["video"]).start()
    fileStream = True
    vs = VideoStream(src=0).start()
    # vs = VideoStream(usePiCamera=True).start()
    fileStream = False
    time.sleep(1.0)

    while True:
        if fileStream and not vs.more():
            break

        frame = vs.read()
        frame = imutils.resize(frame , width = 600)
        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

        rects = detector(gray , 0)

        for rect in rects:
            shape = predictor(gray , rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lstart:lend]
            rightEye = shape[rstart:rend]
            leftear = ear(leftEye)
            rightear = ear(rightEye)

            EAR = (leftear+rightear)/2

            lul = cv2.convexHull(leftEye)
            rul = cv2.convexHull(rightEye)

            cv2.drawContours(frame , lul , -1 , (0,0,255) , 2)
            cv2.drawContours(frame , rul , -1 , (0,255,0) , 2)

            if EAR < EAR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EAR_FRAMES:
                    TOTAL+=1

                COUNTER=0

            cv2.putText(frame , "Blinks: {}".format(TOTAL) , (10,30) , cv2.FONT_HERSHEY_SIMPLEX , 0.7 , (255,0,0) , 2)
            cv2.putText(frame , "EAR: {:.2f}".format(EAR) , (300,30) , cv2.FONT_HERSHEY_SIMPLEX , 0.7 , (255,0,0) , 2)
        
        cv2.imshow("Frame" , frame)
        key = cv2.waitKey(1) & 0XFF

        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    vs.stop()