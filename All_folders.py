import os

si = [x[0] for x in os.walk('./input/new_input/train')]
all = []
with open('folders_name.txt', 'w') as file1:
    file1.write('{')
    counter = -1
    for i in si:
        tak = i.split("\\")
        file1.write(str(counter))
        file1.write(": '" + tak[-1] + "'" + ", ")
        counter += 1
    file1.write('}')