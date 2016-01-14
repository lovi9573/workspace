'''
Created on Jan 13, 2016

@author: jlovitt
'''

N_COLUMNS = 3
N_STEPS = 2
LAYERS = [Layer()]

def converged(a, b):
  if a == None or b == None:
    return False
  else:
    return a == b



if __name__ == '__main__':
  dp = LMDBDataProvider()
  imgkeys = dp.get_keys()
  columns = {}
  for i in range(N_COLUMNS):
    columns[i] = make_column()
  immap = map_img_2_col(imgkeys, columns)
  for l in range(2):
    immap_old = None
    for column in columns.values():
      column.add_layer(LAYERS[l])
    while(not converged(immap, immap_old)):
      encode(immap, N_STEPS)
      immap_old = immap
      immap = map_img_2_col(imgkeys, columns)