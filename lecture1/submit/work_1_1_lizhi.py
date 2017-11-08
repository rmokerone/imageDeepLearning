# -*- coding: utf-8 -*-
list=[]
with open('C:\\work\dataset_1_1.txt','r') as f:
    for l in f.readlines():
        new=l.strip()
        L=new.split(' ')
        list.append(int(L[0])+int(L[1]))
    print(list)
with open('C:\\work\output_1_1.txt','w') as g:
    for S in list:
        g.write(str(S)+'\n')