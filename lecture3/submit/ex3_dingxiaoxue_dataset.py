f1=open('c:\\dataset_2_1.txt','r')
f2=open('c:\\dataset_2_2.txt','w')
for line in f1.readlines():
    l=line.split(',')
    f2.write(str(float(l[0]))+',')
    f2.write(str(float(l[1]))+',')
    f2.write(str(float(l[0])+float(l[1]))+'\n')
f1.close()
f2.close()

