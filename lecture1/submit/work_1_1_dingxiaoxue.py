f=open('/Users/Administrator/dataset_1_1.txt','r')
output=open('/Users/Administrator/output_1_1.txt','w')
for line in f.readlines():
    l=[]
    l=line.split()
    output.write(str(int(l[0])+int(l[1]))+'\n')
f.close()
output.close()

