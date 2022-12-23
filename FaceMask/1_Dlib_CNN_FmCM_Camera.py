import numpy as np
import dlib
import cv2 as cv
# load json and create model
from keras.models import model_from_json
import imutils
import time



json_file = open("FaceMaskDetection_model_file_MobileNet.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("FaceMaskDetection_model_file_MobileNet.h5")
print("Loaded model from disk")


mask_label = {0:'NO MASK',1:'MASK'}

capture = cv.VideoCapture(0)

dlib_detector = dlib.get_frontal_face_detector()
#dlib_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
while True:
    boolean, frame = capture.read()
    if boolean == True:
        
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        coordinate_list = dlib_detector(gray)
        for i, face in enumerate(coordinate_list):
            try:
                x1,y1,x2,y2 = face.left(), face.top(),face.right(),face.bottom()
                #x1,y1,x2,y2 = face.rect.left(), face.rect.top(),face.rect.right(),face.rect.bottom()
                #print("face:",i," >> ",x1,y1,x2,y2)

                ROI = frame[y1:y1+224, x1:x1+224]
                ROI = imutils.resize(ROI, width = 224 )
                if (x2-x1<y2-y1):
                    ROI = frame[y1:y2, x1:x1+y2-y1]
                    ROI = imutils.resize(ROI, width = 224 )
                elif (y2-y1 <224):
                    ROI = frame[y1:y1+x2-x1, x1:x2]
                    ROI = imutils.resize(ROI, height=224 )

                ROI = np.reshape(ROI, (1, 224, 224,3))
                face_mask = np.reshape(ROI,[1,224,224,3])
                face_mask = ROI/255.0
                label = np.argmax(loaded_model.predict(face_mask, verbose=0))
                title= mask_label[label]


                cv.rectangle(frame,(x1,y1), (x2, y2), (0,255,0), 2)
                cv.putText(frame,title, (x1,y1), cv.FONT_HERSHEY_COMPLEX,1,(244,250,250),2)
            except:
                pass
    
            cv.imshow ("live face detection", frame) 
        
        if cv.waitKey(20) == ord('x'):
            break
        #time.sleep(1.5)
capture.release()
cv.destroyAllWindows()