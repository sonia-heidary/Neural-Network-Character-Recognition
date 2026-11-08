
def input_file(path:str):
    """
    :param path: a path to a file which contains the paths to other files. each of the paths belongs to a text file in
    which there is an input for teaching the machine
    :return: a list of paths
    """
    files = []
    with open(path, "r") as f:
        for line in f:
            files.append(line.strip("\n"))
    return files


def read_teaching_input(path: str):
    """
    :param path: this is a path to a file which contains an entry to teach the neural network
    keep in consideration that the first line of the file is the value of target
    :return: a list that contains a list and int --> [[], int] --> [list of elements, target]
    """
    shape = []
    with open(path, 'r') as f:
        target = int(f.readline()) # catch the error
        for line in f:
            line = line.strip("\n")
            line = list(line)
            for element in line:
                shape.append(variable_structuring(element))

    return [shape, target]

def variable_structuring(var: str):
    if var == "#":
        return 1
    elif var == ".":
        return -1
    else:
        return 0