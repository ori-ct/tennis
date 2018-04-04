#!env/bin/python3

import urllib.request
import json
import sys
import numpy as np
import os

def progress_bar(iteration, total, barLength=50):
    percent = int(round((iteration / total) * 100))
    nb_bar_fill = int(round((barLength * percent) / 100))
    bar_fill = '#' * nb_bar_fill
    bar_empty = ' ' * (barLength - nb_bar_fill)
    sys.stdout.write("\r  [{0}] {1}%".format(str(bar_fill + bar_empty), percent))
    sys.stdout.flush()
    if iteration==total:
        print('\n')

def download_image(location,zoom,imsize,scale,fname,APIkey):
    url = 'https://maps.googleapis.com/maps/api/staticmap?'
    url += 'center='+str(location[0])+','+str(location[1])
    url += '&zoom='+str(zoom)
    url += '&size='+str(imsize[0])+'x'+str(imsize[1])
    url += '&scale='+str(scale)
    url += '&maptype=satellite&key='+APIkey
    with urllib.request.urlopen(url) as response, open(fname,'wb') as out_file:
        data = response.read()
        out_file.write(data)

if len(sys.argv)==7:
    min_lat = float(sys.argv[1])
    min_lon = float(sys.argv[2])
    max_lat = float(sys.argv[3])
    max_lon = float(sys.argv[4])
    scene_name = sys.argv[5]
    APIkey = sys.argv[6]
    try:
        os.stat(scene_name)
    except:
        os.mkdir(scene_name) 
else:
    
    if sys.argv[1]=='-h':
        print('Usage:')
        print('   ' + sys.argv[0] + ' <minimum_latitude> <minimum_longitude> <maximum_latitude> <maximum_longitude> <scene_name> <google_maps_API_key>')
    else:
        print('Unexpected number of arguments.')
        print(sys.argv[0] + ' -h')
    exit()

zoom =18
imsize = (256,256)#(1024,1024)
scale = 2
step_lat = 0.0009765625*0.9/2;
step_lon = 0.0009765625*1.4/2;
v1 = np.arange(min_lat,max_lat,step_lat)
v2 = np.arange(min_lon,max_lon,step_lon)
print(str(len(v2)*len(v1)) + ' tiles staged for downloading.')
print('Total of ' + str(len(v2)*len(v1)*0.2 ) + ' Mb will be downloaded. Continue? [Y/n]')
if input()=='n':
     exit()
print('downloading...')

for t1,lat in zip(range(len(v1)),v1):
    progress_bar(t1, len(v1), barLength=20) 
    for t2,lon in zip(range(len(v2)),v2):
        fname = scene_name + '/tile_'+str(t1)+'_'+str(t2)+'.png'
        download_image((lat,lon),zoom,imsize,scale,fname,APIkey)




