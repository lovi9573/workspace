'''
Created on Mar 8, 2016

@author: jlovitt
'''
import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt



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


if __name__ == '__main__':

  path = sys.argv[1]
  print(os.listdir(path))
  #get directory listing
  files = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path,x))]
  print(files)
  #split files by mean length
  files_by_tag = {}
  for imgfile in files:
    tag = imgfile[0:len('generated00')].strip("0")
    if len(tag) == len('generated0'):
      tag = tag[0:len('generated')]+'0'+tag[-1]
    if tag not in  files_by_tag:
      files_by_tag[tag] = []
    files_by_tag[tag].append(imgfile)
  print(files_by_tag)
  results = {}
  for tag,file_list in files_by_tag.iteritems():
    if tag not in results:
      results[tag] = []
    for imgfile in file_list:
      print(imgfile)
      results[tag].append(harrisedge(imgfile))
    
#       cv2.imshow('dst',img)
#       if cv2.waitKey(0) & 0xff == 27:
#           cv2.destroyAllWindows()
  orderedtags = sorted(results.keys())
  m = []
  v = []
  for tag in orderedtags:
    data = np.array(results[tag])
    print("================="+tag+"=================")
    m.append(np.mean(data))
    v.append(np.var(data))
    print(np.mean(data))
    print(np.var(data))
  plt.plot(m)
  plt.show()
  plt.plot(v)
  plt.show()
    
    
    
    
    