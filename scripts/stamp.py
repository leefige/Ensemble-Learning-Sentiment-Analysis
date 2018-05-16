#!/usr/bin/env python3
#-*- coding: UTF-8 -*-

import sys

def writeStamp(no):
    fn = "stamp_" + str(no)
    with open(fn, 'w') as fout:
        fout.write("stamp_" + str(no) + "\n")

if __name__ == '__main__':
    writeStamp(sys.argv[1])
        