import os
#os.getcwd()
#os.chdir('C:\\Users\Administrator\AppData\Local\Programs\lecture1')
f=open('dataset_1_1.txt','r')
g=open('output_1_1.txt','w+')
while True:
    x=f.readline()
    if x=='':
        break
    i=0
    s=0
    n=0
    d=len(x)
    while x[i]!=' ':
        s=s*10+int(x[i])
        i=i+1
    i=i+1
    while i<(d-1):
        n=n*10+int(x[i])
        i=i+1
    g.write(str(s+n)+'\n')
f.close()
g.close()












