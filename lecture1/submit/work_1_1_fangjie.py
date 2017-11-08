def itemfactory(filename,parserfunc):
    with open(filename,'rt') as handle:
        for Ln in handle:
            yield parserfunc(Ln)
with open('C:/work/output_1_1.txt','w') as f:
    for a,b in itemfactory('C:/work/dataset_1_1.txt',lambda Ln: map(int,Ln.split())):
        num=a+b
        f.write(str(num)+'\n')
        print(num)

    
