import cv2
from cv2 import THRESH_BINARY
from tracker import *
tracker = EuclideanDistTracker()
cap= cv2.VideoCapture("chip-video.avi")

object_detector= cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=50)
#object_detector here is extract the moving objects from the video



while True: #loop for extract the frames one after another

    ret, frame= cap.read()
    #object detection

    mask=object_detector.apply(frame) #we want to apply the detection on the frame so we put the frame into the parathesis in here


    _,mask=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    contours, _=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    detections=[]
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area > 50:
            #cv2.drawContours(frame,[cnt],-1,(0,255,0),2)
            x,y,w,h=cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255),3)
            detections.append([x,y,w,h])

    #object tracking

    boxes_ids=tracker.update(detections)
  
    for box_id in boxes_ids:
        x,y,w,h,id=box_id
        cv2.putText(frame,str(id),(x,y-15),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255),3)
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask) #showing the mask
        #the aim of the mask is detecting only the object we want in white, other things we want to avoid are black
         

    key= cv2.waitKey(30)
   
    if key==27: #esc on the keyboard
        break
cap.release()
cv2.destroyAllWindows()