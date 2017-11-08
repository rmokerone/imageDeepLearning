
if __name__ == '__main__':
    with open('../lecture1/dataset_1_1.txt', 'r') as inputFile:
        with open('../lecture1/output_1_1.txt', 'w') as outputFile:
            lines = inputFile.readlines()
            for line in lines:
                line = line[:-1].split()
                ret = int(line[0]) + int(line[1])
                outputFile.write('%d\n' % ret)
