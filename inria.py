#
# Module to load the INRIA person dataset (http://pascal.inrialpes.fr/data/human/)
#
# Authors: Prithvijit Chakrabarty, John Kaushagen
#

import os
import cv2
import random
import numpy as np
from ast import literal_eval

#Path to dataset
PATH = '/home/prithvi/dsets/INRIAPerson/'
#Load as grayscale images
COL = False

#Labels for classes
POS = 1         #Person
NEG = 0         #Non-person
SIZE = (64,128) #Input image size

#Feature extractor
hog = cv2.HOGDescriptor()

#Method to extract features and convert to input vector
def feat(img):
    return hog.compute(img).flatten()

#Method to diplay an image
def disp(lab,img):
    cv2.imshow(lab,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Method to read image from file
def read(path):
    img = cv2.imread(path)
    if COL == False:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

#Method for gamma correction
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 
    for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(image, table)

#Method to load the dataset
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
        if COL == False:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        nds.append((cv2.resize(img,SIZE),NEG))
        #Add randomly cropped subimages to the dataset
        for sx,sy in [(8,16),(32,64)]:
            x = int(random.random()*(img.shape[1]-sx))
            y = int(random.random()*(img.shape[0]-sy))
            nds.append((cv2.resize(img[y:y+sy,x:x+sx],SIZE),NEG))
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
