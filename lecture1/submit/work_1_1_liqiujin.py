f = open('dataset_1_1.txt', 'r')
output = open('output_1_1.txt', 'w')
data = f.readlines()
for element in data:
    ele = element.split()
    a1 =int(ele[0])
    a2 =int(ele[1]) 
    c = a1 + a2
    print('c = ', c)
    output.write(str(c)+'\n')
f.close()
output.close()
