import numpy as np
import cv2 

net=cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
classes=[]

#reading all the clsses
with open('names.txt','r') as f:
    classes=[line.strip() for line in f]

#list containing all layer names
layernames=net.getLayerNames()
#i will be output layer position for each iteration(-1 for index)
#using that index to get output layers from list of all layers
out_layers=[layernames[i[0]-1] for i in net.getUnconnectedOutLayers()]
img=cv2.imread('highway.jpg')
height,width,channels = img.shape

#getting 3 channels in grayscale
blob=cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,False)

net.setInput(blob)
outs=net.forward(out_layers)

boxes=[]
confidences=[]
classe=[]

#getting the info about detected objects
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id=np.argmax(scores)
        confidence=scores[class_id]
        if confidence>0.5:
            #getting the coordinates according to original image size
            center_x=int(detection[0]*width)
            center_y = int(detection[1]*height)
            w=int(detection[2]*width)
            h=int(detection[3]*height)
            
            #calculating coordinates for rectangle
            x=int(center_x- w/2)
            y=int(center_y - h/2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            classe.append(classes[class_id])

#Using non max supression to remove the same object detected twice
indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)

for i in range(len(boxes)):
    #indexes contain index of non repeted objects
    if i in indexes:
        x,y,w,h=boxes[i]
        label=classe[i]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.putText(img,label,(x-2,y-2),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)


cv2.imshow('Detection', img)
cv2.waitKey(0)