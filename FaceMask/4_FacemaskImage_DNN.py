import numpy as np
import cv2 as cv
from cv2 import dnn
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

inWidth = 300
inHeight = 300
confThreshold = 0.5

prototxt = './dnn/deploy.prototxt'
caffemodel = './dnn/res10_300x300_ssd_iter_140000.caffemodel'

net = dnn.readNetFromCaffe(prototxt, caffemodel)



frame = cv.imread('./testdata/1.jpeg')
#frame = imutils.resize(frame, width = frame.shape[1]*2)
cols = frame.shape[1]
rows = frame.shape[0]
net.setInput(dnn.blobFromImage(frame, 1.0, (rows, cols), (104.0, 177.0, 123.0), False, False))
detections = net.forward()

perf_stats = net.getPerfProfile()

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > confThreshold:
        xLeftBottom = int(detections[0, 0, i, 3] * cols)
        yLeftBottom = int(detections[0, 0, i, 4] * rows)
        xRightTop = int(detections[0, 0, i, 5] * cols)
        yRightTop = int(detections[0, 0, i, 6] * rows)
        

        try:
            x1,x2,y1,y2= xLeftBottom,xRightTop,yLeftBottom,yRightTop
            ROI = frame[y1:y1+224, x1:x1+224]
            if (x2-x1<y2-y1):
                ROI = frame[y1:y2, x1:x1+y2-y1]
                ROI = imutils.resize(ROI, width = 224 )
            elif (y2-y1 <224):
                ROI = frame[y1:y1+x2-x1, x1:x2]
                ROI = imutils.resize(ROI, height=224 )
                
                
            #print("ROI shape: ", ROI.shape)
            ROI = np.reshape(ROI, (1, 224, 224,3))
            face_mask = np.reshape(ROI,[1,224,224,3])
            face_mask = face_mask/255.0
            label = np.argmax(loaded_model.predict(face_mask, verbose=0))
            title= mask_label[label]
            
            label =title
            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),(0, 255, 0))
            cv.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (xLeftBottom, yLeftBottom),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        except Exception as ex:
            #print(ex)
            pass

cv.imwrite("output_result.png", frame)