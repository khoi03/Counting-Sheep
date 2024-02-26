import cv2 as cv
import numpy as np

def detecting_area(frame):
    h,w,_ = frame.shape
    color = (128,128,128)
    pts = np.array([[0,650], [600,500], [700,400], [1400,450], [1300,1050], [0,1050]]) #sheeps2
    # pts = np.array([[400,550], [5,700], [5,h-5], [w-5,h-5], [w-5,550]]) #off ewe
    mask = np.zeros_like(frame)
    cv.fillPoly(mask, [pts], (1,1,1))

    img_masked = frame * mask
    # cv.imshow("test",img_masked)
    # cv.waitKey(0)
    return img_masked

def draw_bounds(frame):
    h,w,_ = frame.shape
    color = (0,255,255)
    #sheeps2
    frame = cv.line(frame, (0,650), (600,500), color, 3) #tp1 
    frame = cv.line(frame, (600,500), (700,400), color, 3) #tp2
    frame = cv.line(frame, (0,1050), (1300,1050), color, 3) #bp1
    frame = cv.line(frame, (1300,1050), (1400,450), color, 3) #bp2
    frame = cv.line(frame, (0,650), (0,1050), color, 3) #linked line 1
    frame = cv.line(frame, (600,500), (1300,1050), color, 7) #linked line 2
    frame = cv.line(frame, (700,400), (1400,450), color, 3) #linked line 3
    #Off ewe go sheep sorting
    # frame = cv.line(frame, (400,550), (5,700), color, 3) #lp1 (left)
    # frame = cv.line(frame, (5,700), (5,h-5), color, 3) #lp2
    # frame = cv.line(frame, (0,h-5), (w-5,h-5), color, 3) #rp1 (right)
    # frame = cv.line(frame, (w-5,h-5), (w-5,550), color, 3) #rp2
    # frame = cv.line(frame, (w-5,550), (400,550), color, 3) #linked line 1
    
    color = (221, 218, 250)
    pts = np.array([[0,650], [600,500], [700,400], [1400,450], [1300,1050], [0,1050]]) #sheeps2
    # pts = np.array([[400,550], [5,700], [5,h-5], [w-5,h-5], [w-5,550]]) #off ewe
    mask = np.zeros((frame.shape[0],frame.shape[1]))
    cv.fillPoly(mask, [pts], (1,1,1))
    temp = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    temp[mask == 1] = color
    
    frame[mask==1] = 0.7*frame[mask==1] + 0.3*temp[mask==1]
    # cv.imshow("test",frame)
    # cv.waitKey(0)
    
    return frame