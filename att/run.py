import numpy as np
import os
import glob

dirs = [1, 4, 5, 6]

for d in dirs:
    os.system('bash new_test.sh 2 ../../data/eval'+str(d)+' ../../data/eval'+str(d)+'_mv')
    files = glob.glob('output/iter10000/images/*_iter10.png')
    for i, f in enumerate(files):
        os.system('mv '+f+' temp/output_'+str("%04d" % i)+'.png')
    os.system('ffmpeg -framerate 25 -i temp/output_%04d.png -c copy temp/video_'+str(d)+'.mp4')
    os.system('rm temp/*.png')
