import cPickle as pickle
import sys

import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

import inria

model = './human_model.pkl'

FSIZE = (300,300)

NUM_INTRO_FRAMES = 260


def train(grid_search=False, verbose=False):
    (X,Y,vX,vY) = inria.load()
    if verbose:
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



def video(write_video=True):
    classifier = pickle.load(open(model,'rb'))
    background_subtractor = cv2.createBackgroundSubtractorKNN()
    vc = cv2.VideoCapture('/Users/jkausha/Projects/Support-Vector-Networks/datasets/AVSS_AB_Easy_Divx.avi')
    if write_video:
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        out = cv2.VideoWriter('output.avi',fourcc, 25, FSIZE)
    if vc.isOpened():
        for i in range(NUM_INTRO_FRAMES):
            rval,frame = vc.read()
        if inria.COLOR == False:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        rval = False
    fcount = 1
    while rval:
        frame = cv2.resize(frame,FSIZE)
        foreground = background_subtractor.apply(frame)
        kernel = np.ones((3,3), np.uint8)
        foreground = cv2.dilate(foreground, kernel, iterations=2)
        foreground = cv2.medianBlur(foreground, 3)
        foreground = cv2.threshold(foreground,125,255,cv2.THRESH_OTSU)[1]
        mask = foreground.copy()
        __,contours,hierarchy = cv2.findContours(foreground,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        px = []
        loc = []
        for c in contours:
            (x,y,w,h) = cv2.boundingRect(c)
            if (w > 5) & (h > 5):
                px.append(inria.feat(cv2.resize(frame[y:y+h,x:x+w],inria.SIZE)))
                loc.append(((x,y),(x+w,y+h)))
        if len(px) > 0:
            py = classifier.predict(np.array(px))
            for y,l in zip(py,loc):
                if y == 1:
                    cv2.rectangle(frame,l[0],l[1],(255,255,255),3)

        #Display frame
        cv2.imshow('input',frame)
        cv2.imshow('foreground',foreground)
        cv2.imshow('mask',mask)
        #Add frame to frame history
        #frame_arr.append(frame)
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        out.write(frame)
        #Read new frame
        rval,frame = vc.read()
        if inria.COLOR == False:
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


def frames_to_video(frame_arr,outputpath,fps):
    size = (640,480)
    for img in frame_arr:
        img = cv2.resize(img,size)
    fourcc = cv2.cv.CV_FOURCC('D', 'I', 'V', 'X')
    out = cv2.VideoWriter(outputpath,fourcc, fps, size)
    for i in range(len(frame_arr)):
        out.write(frame_arr[i])
    out.release()

def plot_boxes(foreground,frame):
    #return foreground
    kernel = np.ones((35,35), np.uint8)
    foreground = cv2.dilate(foreground, kernel, iterations=2)
    contours,hierarchy = cv2.findContours(foreground,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        (x,y,w,h) = cv2.boundingRect(c)
        if (w > 5) & (h > 5):
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),5)
    return frame

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()

    elif sys.argv[1] == 'video':
        video()

    else:
        print 'Invalid argument'
