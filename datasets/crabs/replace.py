import os

for filename in os.listdir(os.getcwd()):
    if filename.endswith(".txt"):
            with open('{}' .format(filename), 'r+') as file:
                lines = file.readlines()
                file.seek(0, 0) #set the pointer to 0,0 cordinate of file
                for line in lines:
                    row = line.strip().split(" ")

                    if int(row[0]):
                        row[0] = '0'
                        print(row)
                        file.write(" ".join(row) + "\n")