#
# Module to track humans with SVMs.
# This module trains an binary classification model to identify humans.
# The classifier can be run on a video 
# 
# Authors: Prithvijit Chakrabarty, John Kaushagen

import cv2
import sys
import numpy as np
import cPickle as pickle
from sklearn.svm import SVC
from sklearn.externals import joblib

#Module to load the INRIA person dataset: See inria.py
import inria

#Path to save the model
model = './human_model.pkl'

#Frame sizes
FSIZE = (300,300)   #Display frame size
OUTSIZE = (640,480) #Output frame size

#Parameters for postprocessing background subtraction
#Note: These might be application dependent - may need tuning
LRATE = 0.1         #Set to -1 to keep default
MORPH_KER = (7,7)   #Shape of kernel for dilation
DILATE_ITER = 2     #Number of iterations of dilation
BLUR_KER = (5,5)    #Size of kernel for blurring

#Minimum contour size
MIN_W = 5
MIN_H = 5

#Output video file
VID_INPUT = 'camera'        #Set this to the path if not using camera
#VID_INPUT = '<path_to_video_file>'
VID_OUTPUT = './output.avi'

if sys.argv[1] == 'train':
    #print 'Loading dataset...'
    (X,Y,vX,vY) = inria.load()
    #Train
    print 'Training model...'
    clf = SVC(C=5.0,gamma=0.01,kernel='rbf',probability=True)
    clf.fit(X,Y)
    print 'Done!'
    #Save model
    print 'Saving to disk...'
    pickle.dump(clf,open(model,'wb'))
    print 'Done!'
    #Load model
    print 'Loading model...'
    clf = pickle.load(open(model,'rb'))
    print 'Model loaded'
    #Validation
    print 'Validation...'
    Y_ = clf.predict(vX)
    v = (Y_ - vY)
    v[v!=0] = 1
    print 'Validation accuracy:',(1-np.sum(v)/float(np.size(v)))

elif sys.argv[1] == 'video':
    clf = pickle.load(open(model,'rb'))
    fgbg = cv2.BackgroundSubtractorMOG()
    
    if VID_INPUT == 'camera':
        vc = cv2.VideoCapture(0) #Use this for camera input
    else:
        vc = cv2.VideoCapture(VID_INPUT)
    
    fourcc = cv2.cv.CV_FOURCC(*"DIVX")
    out = cv2.VideoWriter(VID_OUTPUT, fourcc, 25, OUTSIZE)
    
    if vc.isOpened():
        #fps = vc.get(cv2.cv.CV_CAP_PROP_FPS)
        #print fps
        for i in range(260):
            rval,frame = vc.read()
        if inria.COL == False:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        rval = False
    fcount = 1
    while rval:
        frame = cv2.resize(frame,FSIZE)
        if LRATE == -1:
            fg = fgbg.apply(frame)
        else:
            fg = fgbg.apply(frame,learningRate=LRATE)

        #Post-process after background subtraction
        kernel = np.ones(MORPH_KER, np.uint8)               #Morphology kernel
        fg = cv2.dilate(fg, kernel, iterations=DILATE_ITER) #Dilate: connect the contours
        fg = cv2.GaussianBlur(fg,BLUR_KER,0)                #Blur: Smoothen the edges after dilate
        fg = cv2.threshold(fg,125,255,cv2.THRESH_OTSU)[1]   #Threshold blurred image back to binary image
        mask = fg.copy()
        contours,hierarchy = cv2.findContours(fg,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        px = []
        loc = []
        for c in contours:
            (x,y,w,h) = cv2.boundingRect(c)
            #Ignore really small contours
            if (w > MIN_W) & (h > MIN_H):
                px.append(inria.feat(cv2.resize(frame[y:y+h,x:x+w],inria.SIZE)))
                loc.append(((x,y),(x+w,y+h)))
        if len(px) > 0:
            #Convert to a color frame: Get colored bounding boxes
            frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
            #Run SVM predict on contours
            py = clf.predict(np.array(px))
            for y,l in zip(py,loc):
                if y == 1:
                    cv2.rectangle(frame,l[0],l[1],(0,255,0),2)

        #Display frame
        cv2.imshow('input',frame)
        cv2.imshow('fg',fg)
        cv2.imshow('mask',mask)
        
        #Save frame to file
        frame = cv2.resize(frame,OUTSIZE)
        out.write(frame)
        
        #Read new frame
        rval,frame = vc.read()
        if inria.COL == False:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        fcount += 1
        #Exit on escape
        key = cv2.waitKey(10)
        if key == 27:
            break
    #Save frame history
    vc.release()
    out.release()
    cv2.destroyAllWindows()
