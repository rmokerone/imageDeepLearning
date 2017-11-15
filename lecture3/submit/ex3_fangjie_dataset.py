f1 = open('dataset_1_1.txt')
f2 = open('output_1_1.txt')
f3 = open('1.txt','w')
lines1 = f1.readlines()
lines2 = f2.readlines()
for line1 in lines1:
    k = line1.split()[0]
    p = line1.split()[1]
    l = int(line1.split()[0]) + int(line1.split()[1])
    f3.write(str(k)+ ' ' +str(p) + ' ' + str(l) + '\n')
f1.close()
f2.close()
f3.close()
