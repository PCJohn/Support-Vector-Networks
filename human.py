import cPickle as pickle
import sys

import cv2
import numpy as np

from sklearn.svm import SVC
from sklearn.externals import joblib

import inria

model = './human_model.pkl'
#model = './clf.pkl'

VIDEO_PATH = '/Users/jkausha/Projects/Support-Vector-Networks/datasets/AVSS_AB_Easy_Divx.avi'
FSIZE = (300, 300)


def main():
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'video':
        do_video()

def frames_to_video(frame_arr, outputpath, fps):
    size = (640, 480)
    for img in frame_arr:
        img = cv2.resize(img, size)
    fourcc = cv2.cv.CV_FOURCC('D', 'I', 'V', 'X')
    out = cv2.VideoWriter(outputpath, fourcc, fps, size)
    for frame in frame_arr:
        out.write(frame)
    out.release()

# Unsused atm
# def plot_boxes(fg, frame):
#     #return fg
#     kernel = np.ones((35, 35), np.uint8)
#     fg = cv2.dilate(fg, kernel, iterations=2)
#     #return fg

#     """mask = np.zeros(fg.shape).astype(np.float32)
#     contours,hierarchy = cv2.findContours(fg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     for c in contours:
#         (x,y,w,h) = cv2.boundingRect(c)
#         if (w > 0) & (h > 0):
#             mask[y:y+h,x:x+w] += 1
#     mx = np.max(mask)
#     if mx > 0:
#         mask /= mx
#     mask = np.uint8(255*mask)
#     #mask = cv2.GaussianBlur(mask,(35,35),0)
#     ret2,mask = cv2.threshold(mask,125,255,cv2.THRESH_OTSU)"""

#     contours,hierarchy = cv2.findContours(fg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     for c in contours:
#         x, y, w, h = cv2.boundingRect(c)
#         if (w > 5) & (h > 5):
#             cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 5)
#     return frame

def train():
    X, Y, vX, vY = inria.load()

    #Train
    print 'Training model...'
    clf = SVC(C=5.0, gamma=0.01, kernel='rbf', probability=True)
    clf.fit(X, Y)
    print 'Done!'
    #Save model
    print 'Saving to disk...'
    pickle.dump(clf, open(model, 'wb'))
    print 'Done!'
    #Load model
    print 'Loading model...'
    clf = pickle.load(open(model, 'rb'))
    print 'Model loaded'
    #Validation
    print 'Validation...'
    Y_ = clf.predict(vX)
    v = (Y_ - vY)
    v[v != 0] = 1
    print 'Validation accuracy:', (1-np.sum(v)/float(np.size(v)))


def do_video():
    clf = pickle.load(open(model, 'rb'))
    if cv2.__version__ < '3.0.0':
        fgbg = cv2.BackgroundSubtractorMOG()
    else:
        fgbg = cv2.createBackgroundSubtractorMOG2()
    video = cv2.VideoCapture(VIDEO_PATH)

    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv2.VideoWriter('output.avi', fourcc, 25, FSIZE)

    read_success = False
    if video.isOpened():
        for i in range(260):
            read_success, frame = video.read()
        if inria.COL == False:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fcount = 1
    #frame_arr = []
    while read_success:
        frame = cv2.resize(frame, FSIZE)
        #fg = fgbg.apply(frame)
        fg = fgbg.apply(frame, learningRate=0.01)

        kernel = np.ones((3, 3), np.uint8)
        fg = cv2.dilate(fg, kernel, iterations=2)
        fg = cv2.GaussianBlur(fg, (5, 5), 0)
        fg = cv2.threshold(fg, 125, 255, cv2.THRESH_OTSU)[1]
        mask = fg.copy()
        if cv2.__version__ < '3.0.0':
            contours, hierarchy = cv2.findContours(fg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            __, contours, hierarchy = cv2.findContours(fg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        px = []
        loc = []
        for c in contours:
            (x,y,w,h) = cv2.boundingRect(c)
            if (w > 5) & (h > 5):
                px.append(inria.feat(cv2.resize(frame[y:y+h, x:x+w], inria.SIZE)))
                loc.append(((x, y), (x+w, y+h)))
        if len(px) > 0:
            py = clf.predict(np.array(px))
            for y, l in zip(py, loc):
                if y == 1:
                    cv2.rectangle(frame, l[0], l[1], (255, 255, 255), 3)

        #Display frame
        cv2.imshow('input', frame)
        cv2.imshow('fg', fg)
        cv2.imshow('mask', mask)
        #Add frame to frame history
        #frame_arr.append(frame)
        out.write(frame)
        #Read new frame
        read_success, frame = video.read()
        if inria.COL == False:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fcount += 1
        #Exit on escape
        key = cv2.waitKey(10)
        if key == 27:
            break
    #Save frame history
    #frames_to_video(frame_arr,'output.avi',25)
    video.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
