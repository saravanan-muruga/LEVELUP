# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 21:02:33 2021

@author: saravanan.muruga
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 07:37:17 2021

@author: saravanan.muruga
"""
import cv2 as cv
import numpy as np
import json
class BgModel():
    
    def __init__(self):
        self.backSub = cv.createBackgroundSubtractorMOG2(history=50,varThreshold = 16,detectShadows = False)
        self.kernel = np.ones((3,3), np.uint8)
    
            
        
    def process(self,frame):
        # frame=cv.cvtColor( frame , cv.COLOR_BGR2LAB)
        fgMask = self.backSub.apply(frame)
        img_erosion = cv.erode(fgMask, self.kernel, iterations=1) 
        contours, _ = cv.findContours(image=img_erosion, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE)
        
        detections=[]
        for i in range(len(contours)):
            area = cv.contourArea(contours[i])
            x,y,w,h = cv.boundingRect(contours[i])
            Aspect_ratio = float(w) / h
            ball=frame[y:y+h, x:x+w]
            hist = cv.calcHist([ball],[1],None,[4],[0,256])
  
            # if area > 100 and area<700 and Aspect_ratio> 0.9 and np.argmax(hist) >=2:
            #     cv.rectangle(frame, (x+x_, y+y_), (x+w+x_,y+h+y_), (0,255,255))
            
            #     cv.putText(frame, str(area), (x+x_, y+212), cv.FONT_HERSHEY_SIMPLEX, 2, (0,255,255),2)
            if area > 30 and area<500 :#and Aspect_ratio<1 and (np.argmax(hist)==2):
        
                detections.append([x,y,w,h])
                
               
                # cv.rectangle(frame, (x+x_, y+y_), (x+w+x_,y+h+y_), (0,255,255),2)
        
        return detections,img_erosion
    
    
    
def main():
    vid_path='D:/Saravanan/Data/videos/1.mp4'
    # vid_path='D:/Saravanan/Data/videos/11th december/pitch_1.mp4'
    # vid_path='D:/Saravanan/Data/videos/11th december/batsman_1.MP4'
    # vid_path='D://Saravanan//Data//videos//11th december//batsman_3.mp4'
    video_capture=cv.VideoCapture(vid_path)
    ret, frame = video_capture.read()
    size = frame.shape[:2]
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
       
    size = (frame_width, frame_height)
    video_writer = cv.VideoWriter('slipcam_1.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, size)

    bg_model=BgModel()
    cv.namedWindow("output", cv.WINDOW_NORMAL) 
    cv.namedWindow("mask", cv.WINDOW_NORMAL) 
 
    frame_no=0
    json_path = "D://Saravanan//Data//videos//11th december//batsman_3_bg.json"
    frame_dict = {}
    # tracker = cv.TrackerMIL_create()
    tracker=cv.TrackerCSRT_create()
    i=0
    prev_frames=[]
    rects=[]
    final_rects={}
    # trackeri=False
    while True:
        ret, frame = video_capture.read()

        if frame is None:
            break
        # roi=(454, 169, 546, 348)#pitch 11 dec
        # roi=(633, 489, 324, 217)# batsman cam 11 dec
        
        roi=(977, 202, 389, 369)  # slipcam
        # roi=(407, 212, 793, 396)
        roi_frame=frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        detections,mask=bg_model.process(roi_frame)
        if i==0:
            prev_frames.append(frame)
            
            
        if detections:
            if  i==0:
                detections_=detections[0]

                tracker.init(frame,[detections_[0]+roi[0],detections_[1]+roi[1], detections_[2],detections_[3]])
                
                for c,frames in enumerate(prev_frames[::-1][:30]):
                    
                    ok,bbox=tracker.update(frames)
                    if ok:
                        # Tracking success
                        print('tracking')
                        p1 = (int(bbox[0]), int(bbox[1]))
                        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                        rects.append([int(bbox[0]), int(bbox[1]),int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])])
                        cv.rectangle(frames, p1, p2, (255, 0, 0), 3)
                        f=frame_no-c+1
                        final_rects[f]=[int(bbox[0]), int(bbox[1]),int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]
                     
                i+=1
            else:
                detections=detections[0]
                # if i==2:
                tracker.init(frame,[detections[0]+roi[0],detections[1]+roi[1], detections[2],detections[3]])
                ok,bbox=tracker.update(frame)
                if ok:
                    # Tracking success
                    print('tracking')
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv.rectangle(frame, p1, p2, (255, 0, 0), 3)
                    rects.append([int(bbox[0]), int(bbox[1]),int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])])

                    final_rects[frame_no]=[int(bbox[0]), int(bbox[1]),int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]

                    
                cv.rectangle(frame, (detections[0]+roi[0],detections[1]+roi[1]), (detections[0]+roi[0]+detections[2],detections[1]+roi[1]+detections[3]), (0,255,255),3)
                frame_dict[frame_no]=[detections[0]+roi[0],detections[1]+roi[1], detections[2],detections[3]]
        else:

            frame_dict[frame_no]=[]
        
        for j in rects:
            cv.rectangle(frame, (j[0],j[1]), (j[2],j[3]), (255, 0, 0), 3)
            
        
        cv.imshow('output', frame)
        cv.imshow('mask', mask)
        video_writer.write(frame)
        frame_no+=1
        keyboard = cv.waitKey(1)
        if keyboard == 'q' or keyboard == 27:
            cv.destroyAllWindows()
            video_capture.release()
            video_writer.release()
    with open('final_rects_csrt.json', "w") as outfile:
        print('writting json')
        json.dump(final_rects, outfile) 
    cv.destroyAllWindows()
    video_capture.release()
    video_writer.release()


main()
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
