import os
import cv2
import random
import numpy as np
from ast import literal_eval

COL = False
PATH = '/Users/jkausha/Projects/Support-Vector-Networks/datasets/INRIAPerson/'
POS = 1
NEG = 0
SIZE = (64,128)

hog = cv2.HOGDescriptor()

def feat(img):
    return hog.compute(img).flatten()

def disp(lab,img):
    cv2.imshow(lab,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def read(path):
    img = cv2.imread(path)
    if COL == False:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 
    for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(image, table)

def load():
    print 'Loading dataset...'
    #Load positive samples
    ann_path = os.path.join(PATH,'Train/annotations')
    pds = []
    for ant in os.listdir(ann_path):
        ant = os.path.join(ann_path,ant)
        with open(ant,'r') as f:
            for l in f.readlines():
                if l.startswith('Image filename :'):
                    img = cv2.imread(os.path.join(PATH,l.split(':')[1].strip()[1:-1]))
                    if COL == False:
                        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                elif l.startswith('Bounding box'):
                    bbox = (l.split(':')[1].strip()).split(' - ')
                    x1,y1 = literal_eval(bbox[0])
                    x2,y2 = literal_eval(bbox[1])
                    crop = cv2.resize(img[y1:y2,x1:x2],SIZE)
                    pds.append((crop,POS))
    #Load negative samples
    nds = []
    for ant in os.listdir(os.path.join(PATH,'Train/neg')):
        img = cv2.imread(os.path.join(PATH,'Train/neg',ant))
        #disp(ant,img)
        if COL == False:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        nds.append((cv2.resize(img,SIZE),NEG))
        for sx,sy in [(8,16),(32,64)]:
            x = int(random.random()*(img.shape[1]-sx))
            y = int(random.random()*(img.shape[0]-sy))
            nds.append((cv2.resize(img[y:y+sy,x:x+sx],SIZE),NEG))
    print len(pds),len(nds)
    #Shuffle dataset
    random.shuffle(pds)
    random.shuffle(nds)
    train = pds[:1000]+nds[:1000]
    val = pds[1000:1200]+nds[1000:1200]
    X,Y = map(np.array,zip(*train))
    vX,vY = map(np.array,zip(*val))
    print 'Done! Raw dataset shape:',X.shape,Y.shape
    print 'Applying feature extractor...'
    X = np.array(map(feat,list(X)))
    vX = np.array(map(feat,list(vX)))
    print 'Done! Feature dataset shape:',X.shape,Y.shape
    return (X,Y,vX,vY)

#X,Y = load()
#for x,y in zip(X[:5],Y[:5]):
#    disp(str(y),x)
