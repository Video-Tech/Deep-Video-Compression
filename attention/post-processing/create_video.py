import os
import glob
import numpy as np

vids = {'338',
        '407',
        '352',
        '509',
        '554',
        '039',
        '606',
        '363',
        '629',
        '406',
        '368'}

for vid in vids:
    files = glob.glob('../no-att/output/iter50000/images/*_'+vid+'_*.png')
    for i, f in enumerate(sorted(files)):
        os.system('cp '+f+' temp/output_'+str("%04d" % i)+'.png')
    os.system('ffmpeg -framerate 25 -i temp/output_%04d.png -c copy temp/video_'+vid+'.mp4')
    os.system('rm temp/*.png')
