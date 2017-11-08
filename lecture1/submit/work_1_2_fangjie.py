l=list((open('C:/work/dataset_1_2.raw','rb')).read())
a=l.count(0)
b=l.count(255)
c=l.count(0)+l.count(255)
print('0的个数为:',a)
print('1的个数为:',b)
print('总的个数为：',c)

    

