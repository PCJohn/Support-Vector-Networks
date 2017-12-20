from __future__ import print_function

import cPickle as pickle
import sys

import cv2
import numpy as np
from sklearn.svm import SVC

import inria


model = './human_model.pkl'


FSIZE = (300,300)
OUTSIZE = (640,480)


if sys.argv[1] == 'train':
    (X_train, Y_train, X_verify, Y_verify) = inria.load()
    print('Training model...')
    clf = SVC(C=5.0,gamma=0.01,kernel='rbf',probability=True)
    clf.fit(X_train,Y_train)
    print('Done!')
    #Save model
    print('Saving to disk...')
    pickle.dump(clf,open(model,'wb'))
    print('Done!')
    #Load model
    print('Loading model...')
    clf = pickle.load(open(model,'rb'))
    print('Model loaded')
    #Validation
    print('Validation...')
    Y_predict = clf.predict(X_verify)
    verification = (Y_predict - Y_verify)
    verification[verification!=0] = 1
    print('Validation accuracy:',(1-np.sum(verification)/float(np.size(verification))))

elif sys.argv[1] == 'video':
    clf = pickle.load(open(model,'rb'))
    fgbg = cv2.createBackgroundSubtractorKNN()
    vc = cv2.VideoCapture('/Users/jkausha/Projects/Support-Vector-Networks/datasets/AVSS_AB_Easy_Divx.avi')
    

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter('output.avi', fourcc, 25, OUTSIZE)
    
    if vc.isOpened():
        for i in range(260):
            rval,frame = vc.read()
        if inria.COL == False:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        rval = False
    fcount = 1
    while rval:
        frame = cv2.resize(frame,FSIZE)
        fg = fgbg.apply(frame)
        kernel = np.ones((3,3), np.uint8)
        fg = cv2.dilate(fg, kernel, iterations=2)
        fg = cv2.GaussianBlur(fg,(5,5),0)
        fg = cv2.threshold(fg,125,255,cv2.THRESH_OTSU)[1]
        mask = fg.copy()
        __, contours,hierarchy = cv2.findContours(fg,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        px = []
        loc = []
        for c in contours:
            (x,y,w,h) = cv2.boundingRect(c)
            if (w > 5) & (h > 5):
                px.append(inria.feat(cv2.resize(frame[y:y+h,x:x+w],inria.SIZE)))
                loc.append(((x,y),(x+w,y+h)))
        if len(px) > 0:
            py = clf.predict(np.array(px))
            for y,l in zip(py,loc):
                if y == 1:
                    cv2.rectangle(frame,l[0],l[1],(255,255,255),3)

        cv2.imshow('input',frame)
        cv2.imshow('fg',fg)
        cv2.imshow('mask',mask)
        
        frame = cv2.resize(frame,OUTSIZE)
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
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
    #frames_to_video(frame_arr,'output.avi',25)
    vc.release()
    out.release()
    cv2.destroyAllWindows()
