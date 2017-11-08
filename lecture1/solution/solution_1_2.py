#!/usr/bin/env python
# coding=utf-8

if __name__ == '__main__':
    allCount = 0
    blackCount = 0
    whiteCount = 0

    inputFile = open('../lecture1/dataset_1_2.raw', 'rb')
    data = inputFile.read()
    for element in data:
        allCount += 1
        if element == 0x00:
            blackCount += 1
        elif element == 0xff:
            whiteCount += 1
        else:
            print('File invalid characters')
    print('%d %d %d' % (allCount, blackCount, whiteCount))
