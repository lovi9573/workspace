import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import re
import math
from operator import mul,add
"""
Input file must be a flat vector of image values.
Assume n images of v pixels.
All n pixel 0's will be consecutive, followed by all n pixel 1's, etc...

"""

def closeCommonFactor(numbera, numberb, target):
    target = int(target)
    d = 0
    while target+d < numbera :
        if  numbera%(target+d) == 0 and (target+d)%numberb == 0:
            return target+d
        d += 1
    d = 0
    while d < target:
        if numbera%(target-d) == 0 and (target+d)%numberb == 0:
            return target-d
        d += 1
    return max(numbera,numberb)
        

def closeFactor(numbera, target):
    target = int(target)
    d = 0
    while target+d < numbera :
        if  numbera%(target+d) == 0 :
            return target+d
        d += 1
    d = 0
    while d < target:
        if numbera%(target-d) == 0 :
            return target-d
        d += 1
    return numbera        
 
def tile_imgs(dat):       
  """
  tiles and displays input as images
  dat: an n,y,x,c array
  
  """
  n,y,x,c = dat.shape
  minval = np.min(dat)
  maxval = np.max(dat)
  print "max: {} min: {}\n".format(maxval,minval)
  
  horizontal_images_in_display_ideal = math.sqrt(n*c)
  if n == 1:
      horizontal_images_in_display = closeFactor(c, horizontal_images_in_display_ideal)
      print "horizontal_images_in_display",horizontal_images_in_display
  elif c == 1:
      horizontal_images_in_display = closeFactor(n, horizontal_images_in_display_ideal)
  else:
      horizontal_images_in_display = closeCommonFactor(n*c, c, horizontal_images_in_display_ideal)
  vertical_images_in_display = n*c/horizontal_images_in_display
  print "Display dimensions in images: ({},{})".format(horizontal_images_in_display, vertical_images_in_display)
  dat = dat.transpose([0,3,1,2]) #n,c,y,x

  imgs = np.ndarray([vertical_images_in_display*(y+1),horizontal_images_in_display*(x+1)],dtype=np.float32)
  imgs[:,:] = 0.0
  for img_row in range(vertical_images_in_display):
      for px_row in range(y):
          for img in range(horizontal_images_in_display):
              imgs[px_row+img_row*(y+1),\
                   img*(x+1):(img+1)*(x+1)-1] = \
              dat[(img_row*horizontal_images_in_display+img)/c,\
                   (img_row*horizontal_images_in_display+img)%c ,\
                   px_row,\
                   :]
              imgs[:,(img+1)*(x+1)-1] = minval
      imgs[px_row+img_row*(y+1)+1,:] = minval
  return imgs



def display(data, isprob=False):
  imgs = tile_imgs(data)
  if isprob:
      plt.imshow(imgs,interpolation='none',vmin=0,vmax=1)  
#   elif c == -1: #TODO(jesselovitt): fix to show 3 color images correctly
#       imgs = imgs.reshape([vertical_images_in_display*(y+1)/3,horizontal_images_in_display*(x+1),3]) 
#       img = mpimg.fromarray(dat.flatten())
#       plt.imshow(img)
#       plt.show()
  else:          
      plt.imshow(imgs,interpolation='none')
      plt.set_cmap('gray')
      plt.colorbar()
      plt.show()

fig = None
im = None

def display_animate(data,isprob=False):
    global fig
    global im
    plt.isinteractive()
    imgs = tile_imgs(data)
    if not fig:
      fig = plt.figure()
      im = plt.imshow(imgs)
      plt.set_cmap('gray')
      plt.colorbar()
    else:
      im.set_data(imgs)
    plt.show(block=False)


if __name__ == "__main__":
  for i in range(10):
    x = np.random.randn(16,10,10,1)
    display_animate(x)



#     isprob = False
#     if ("-p" in sys.argv and len(sys.argv) < 3) or ("-p" not in sys.argv and len(sys.argv) < 2):
#         print "Use: "+sys.argv[0]+" <datafile> [-p] \n\t-p  treat values as probabilities (0,1) "
#         sys.exit()
#     if "-p" in sys.argv:
#         isprob=True
#     dims = None
#     dats = []
#     for fname in sys.argv[1:]:
#         if not fname == "-p":
#           with open(fname, "r") as fin:
#             header = fin.readline()
#             dats.append(fin.read())
#             if not dims:
#               dims = map(int,header.strip().split(" ")) 
#             else:
#               dims[3] += map(int,header.strip().split(" "))[3]
#     
#     x,y,c,n = dims 
#     dat = np.ndarray(reduce(mul,dims),dtype=np.float32)
#     i = 0
#     for datum in dats:
#         #Read data into flat array
#         d = re.sub("\s+",",",datum).strip(",")
#         d = d.split(",")
#         d = map(float,d) 
#         d = np.asarray(d,dtype = np.float32)
#         s = d.shape[0]
#         dat[i:i+s] = d
#         i += s
#     display(dat)
#     
