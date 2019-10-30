
# coding: utf-8

# In[ ]:


import os
import time
import glob
import numpy as np


# In[ ]:


files = glob.glob('logs/*.log')
d = []
for f in files:
    data = np.loadtxt(f)
    for i in data:
        d.append(int(i[4]/(i[5]*1000.0)))
    break


# In[ ]:


os.system('tc qdisc add dev enp0s31f6 root handle 1: tbf rate 5mbit latency 1000ms burst 1540')
for i in d:
    print(i)
    os.system('tc qdisc change dev enp0s31f6 root handle 1: tbf rate '+str(i)+'mbit latency 3000ms burst 1540')
    time.sleep(5)
os.system('sudo tc qdisc del dev enp0s31f6 root handle 1: tbf rate '+str(i)+'mbit latency 1000ms burst 1540')

