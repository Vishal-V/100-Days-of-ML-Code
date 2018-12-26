# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 18:16:27 2018

@author: Vishal

The pickle file has to be using Unix new lines otherwise at least Python 3.4's C 
pickle parser fails with exception: pickle.UnpicklingError: the STRING opcode 
argument must be quoted. I think that some git versions may be changing the 
Unix new lines ('\n') to DOS lines ('\r\n').
"""
import sys

if len(sys.argv[1:]) != 2:
  sys.exit(__doc__)

content = ''
outsize = 0
with open(sys.argv[1], 'rb') as infile:
  content = infile.read()
with open(sys.argv[2], 'wb') as output:
  for line in content.splitlines():
    outsize += len(line) + 1
    output.write(line + '\n')

print("Done. Saved %s bytes." % (len(content)-outsize))