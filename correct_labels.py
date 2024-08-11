with open("EqualDefault/equalFreq_test_H.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    
newlines = [lines[0]]
for line in lines[1:]:
    line = line[:-6]
    line += "L EY0\n" 
    newlines.append(line)
    
with open("EqualDefault/equalFreq_test_H_copy.txt", "w", encoding="utf-8") as f:
    f.writelines(newlines)