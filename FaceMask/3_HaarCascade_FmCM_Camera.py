import numpy as np
import cv2 as cv
import time
# load json and create model
from keras.models import model_from_json
import imutils


json_file = open("FaceMaskDetection_model_file_MobileNet.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("FaceMaskDetection_model_file_MobileNet.h5")
print("Loaded model from disk")

mask_label = {0:'NO MASK',1:'MASK'}
#====================



face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
#capture = cv.imread('./testdata/3.jpeg')
capture = cv.VideoCapture(0)
while True:

    ret, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in face:
        try:
            x1,y1,x2,y2= x,y,x+w,y+h
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
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv.putText(frame,title, (x1,y1), cv.FONT_HERSHEY_COMPLEX,1,(244,250,250),2)
        except Exception as ex:
            #print(ex)
            pass
    cv.imshow("detections", frame)
    if cv.waitKey(1) != -1:
        break
    
    #time.sleep(1.5)
capture.release()
cv.destroyAllWindows()