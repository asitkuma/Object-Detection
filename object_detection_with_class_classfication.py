from unittest import result
from ultralytics import YOLO
import cv2
import cvzone
import math
model=YOLO('./gloabal_folder/yolov8l.pt')#yolov8l.pt,yolov8n.pt,yolov8m.pt speed comparision nano>medium>large
face=cv2.VideoCapture(0)
class_names=['Person','Bicycle','Car','Motorbike','Aeroplane','Bus','Train','Truck','boat','traffic light','fire hydrant','stop sign','Parking meter','Bench','Bird','Cat','Dog','Horse','Sheep','Cow','Elephant','beer','Zebra','giraffe','backpack','Umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wineglass','cup','fork','knife','spoon','bowl','banana','apple','sandwitch','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','dinning table','toilet','tv monitor','laptop','mouse','remote','keyboard','cellphone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddybear','hair drier','toothbrush']
while True:
    ret,frame=face.read()
    result=model(frame,stream=True)
    for i in result:
        boxes=i.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,232),2)
            w,h=x2-x1,y2-y1
            # bbox=int(x1),int(y1),int(w),int(h)
            cvzone.cornerRect(frame,(x1,y1,w,h))
            #confidence.
            conf=box.conf[0]
            confidence=math.ceil((conf)*100)/100
            #class ..
            class_name_finder=int(box.cls[0])
            cvzone.putTextRect(frame,f'{class_names[class_name_finder]} {confidence}',(max(0,x1),max(40,y1-20)),2,2)
    cv2.imshow("window",frame)
    cv2.waitKey(1)
