'''
Created on Mar 8, 2016

@author: jlovitt
'''
import cv2
import numpy as np
import scipy.stats as st
import sys
import os
import matplotlib.pyplot as plt
import skimage.feature as ft


def harrisedge(imgfile):
      
      
      #compute stats
      img = cv2.imread(os.path.join(path,imgfile))
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
      gray = np.float32(gray)
      dst = cv2.cornerHarris(gray,2,3,0.04)
    
      #result is dilated for marking the corners, not important
      dst = cv2.dilate(dst,None)
      
      #print(dst)
      # Threshold for an optimal value, it may vary depending on the image.
      mask = np.zeros(dst.shape)
      mask[dst>0.01*dst.max()]=1
      return np.sum(mask)  

def sift(imgfile):
      #compute stats
      img = cv2.imread(os.path.join(path,imgfile))
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
      sift = cv2.SIFT()
      kp = sift.detect(gray,None)
    
      print(kp)
      
      #print(dst)
      # Threshold for an optimal value, it may vary depending on the image.
      mask = np.zeros(dst.shape)
      mask[dst>0.01*dst.max()]=1
      return np.sum(mask)  

def star(imgfile):
      #compute stats
      img = cv2.imread(os.path.join(path,imgfile))
      # Initiate STAR detector
      star = cv2.FeatureDetector_create("STAR")
      # find the keypoints with STAR
      kp = star.detect(img,None)
      return len(kp) 

def orb(imgfile):
      #compute stats
      img = cv2.imread(os.path.join(path,imgfile))
      # Initiate STAR detector
      orb = cv2.ORB()
      # find the keypoints with STAR
      kp = orb.detect(img,None)
      return len(kp) 

def local_binary_pattern(imgfile):
  '''
  https://spnath-xy27.googlecode.com/hg-history/86e39ff0f7263b0c511f6881740a7a170a1a0b7e/src/python/scikits.image/DOC/html/auto_examples/plot_local_binary_pattern.html
  '''
  METHOD = 'uniform'
  P = 16
  R = 2
  img = cv2.imread(os.path.join(path,imgfile))
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  hist = ft.local_binary_pattern(gray, P, R, METHOD)
  print(hist)


if __name__ == '__main__':

  path = sys.argv[1]
  #get directory listing
  files = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path,x))]
  #split files by mean length
  files_by_tag = {}
  for imgfile in files:
    tag = imgfile[0:len('generated00')]
    if tag not in  files_by_tag:
      files_by_tag[tag] = []
    files_by_tag[tag].append(imgfile)
#   print(files_by_tag)
  results = {}
  for tag,file_list in files_by_tag.iteritems():
    if tag not in results:
      results[tag] = []
    for imgfile in file_list:
      print(imgfile)
      results[tag].append(local_binary_pattern(imgfile))
    
#       cv2.imshow('dst',img)
#       if cv2.waitKey(0) & 0xff == 27:
#           cv2.destroyAllWindows()
  orderedtags = sorted(results.keys())
  m = []
  v = []
  tags = []
  for tag in orderedtags:
    data = np.array(results[tag])
    tags.append(int(tag[-2:]))
    print("================="+tag+"=================")
    m.append(np.mean(data))
    v.append(np.var(data))
    print(np.mean(data))
    print(np.var(data))
  plt.plot(tags,m)
  plt.show()
  plt.plot(tags,v)
  plt.show()
  print("Mean correlation:")
  print(st.pearsonr(tags, m)[0])
    
    
    
    
    