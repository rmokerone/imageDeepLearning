f = open ('dataset_1_2.raw','rb')
data = f.read()
number = 0
score = 0

for x in data:
    #print(x)
    if x==0xFF:
        number = number + 1
    elif x == 0x00:
            score = score + 1
    else:
        continue
print('all = ',number + score)
print('number = ',number)
print('score = ',score)
#print('hi')