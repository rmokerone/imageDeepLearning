f=open('C:\\dataset_1_2.raw','rb')
count1=0
count2=0
count3=0
for line in f.readlines():
    for i in range(len(line)):
	    count3=count3+1
	    if line[i]==0:
		    count1=count1+1
	    else:
		    count2=count2+1
print('0x00总数是',count1,'0xff总数是',count2,'像素数是',count3)
f.close()