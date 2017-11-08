#!/usr/bin/env python
# coding=utf-8

import random

if __name__ == '__main__':
    fp = open('../dataset_1_1.txt', 'w')
    for i in range(1024):
        fp.write("{0} {1}\n".format(random.randint(0, 1024),
                                  random.randint(0, 1024)))
    fp.close()
