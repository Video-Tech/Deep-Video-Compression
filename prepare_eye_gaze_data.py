import numpy as np
import os

idir = 'video/'
odir = 'train/'
gdir = 'train_gaze/'

files = os.listdir('video/')

for f in files:
    os.system('ffmpeg -i '+idir+f+' -vf fps=30 -vframes 97 '+odir+'video_'+f[:-4]+'_%04d.png')
    #gfiles = os.listdir('annotation/0'+f[:-4]+'/maps/')
    #gfiles = ["%04d" % (x+1) for x in range(97)]
    #for gf in gfiles:
    #    os.system('cp annotation/0'+f[:-4]+'/maps/'+gf+'.png '+gdir+'video_gaze_map_'+f[:-4]+'_'+gf+'.png')
