#!/usr/bin/env python
# coding=utf-8

import random

if __name__ == '__main__':
    fp = open('../dataset_1_2.raw', 'wb')
    for i in range(72*72):
        if random.randint(0, 200) % 2:
            fp.write(b'\x00')
        else:
            fp.write(b'\xff')
    fp.close()
