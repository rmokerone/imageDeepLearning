L=[]
with open('dataset_1_2.raw','rb') as f:
    for l in f.read():
	    L.append(l)
    print(L)
s=len(L)
print('总数',s)
i=0
j=0
for k in range(s):
    if L[k]==0:
	    i=i+1
    else:
	    j=j+1
print('0x00数',i)
print('0xFF数',j)
