paths = ["EqualDefault/equalFreq", "MinDefault/minDefault", "MajDefault/majDefault"]

for path in paths: 
    modified_lines = []

    with open(f"{path}_test.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    modified_lines.append(lines[0])

    for line in lines[1:]:
        sg, pl = line.split(",")
        sg = sg.split(" ")[:-1] + ["_"]
        sg = " ".join(sg)
        modified_lines.append(sg + "," + pl)

    with open(f"{path}_test_zeros.txt", "w", encoding="utf-8") as f:
        f.writelines(modified_lines)