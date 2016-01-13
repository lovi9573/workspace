'''
Created on Jan 13, 2016

@author: jlovitt
'''

N_COLUMNS = 3
N_STEPS = 2

if __name__ == '__main__':
    imgs = get_train_data_keys()
    columns = []
    for i in range(N_COLUMNS):
      columns.append(make_column())
    immap = map_img_2_col(imgs, columns)
    immap_old = immap
    while(not converged(immap, immap_old)):
      immap_old = immap
      immap = map_img_2_col(imgs, columns)
      encode(immap, columns, N_STEPS)