#!/usr/local/bin/python3
# Note, if any output has NAN in it, we drop the entire episode from the calculation.

import glob
import re

files = glob.glob('measurements*.txt')
f1 = open('ue_1.txt', 'a')
f2 = open('ue_2.txt', 'a')

pattern = re.compile('[\[\]_ \':a-z]+') # get rid of [], colons, and words.

for file in files:
    f = open(file, 'r')
    lines = f.read()
    sinr1 = lines.split(':')[1]
    sinr2 = lines.split(':')[2]
    
    if ('nan' in sinr1) or ('nan' in sinr2):
        continue
    
    # Clean up sinr1, 2 by replacing pattern with ''
    f1.write('{},'.format(re.sub(pattern, '', sinr1)))
    f2.write('{},'.format(re.sub(pattern, '', sinr2)))

f1.close()
f2.close()
f.close()

