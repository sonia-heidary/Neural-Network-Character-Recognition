import perceptron
import file_handling
import MultiOutputPerceptron
import IndependentPerceptrons


def initialize():
    """
    this function is the core of the program
    :return: the weights of the neural network
    """
    # reading the input files and extracting the inputs for teaching the neural network
    input_files_path_location = "inputs.txt"
    input_files_paths = file_handling.input_file(input_files_path_location)
    inputs = []
    for i in input_files_paths:
        inputs.append(file_handling.read_teaching_input(i))

    # teaching the neural network
    my_perceptron = perceptron.Perceptron(inputs)
    weights = my_perceptron.weight
    weight_bios = my_perceptron.weight_bios
    weights.append(weight_bios)

    return weights

def initialize_multi():
    input_files_path_location = "inputs_Multi.txt"
    input_files_paths = file_handling.input_file(
        input_files_path_location
    )

    inputs = []

    for path in input_files_paths:
        inputs.append(file_handling.read_teaching_input(path))

    my_perceptron = MultiOutputPerceptron.MultiOutputPerceptron(inputs)

    return my_perceptron

def initialize_independent():

    input_files_path_location = "inputs_Multi.txt"

    input_files_paths = file_handling.input_file(
        input_files_path_location
    )

    inputs = []

    for path in input_files_paths:
        inputs.append(
            file_handling.read_teaching_input(path)
        )

    # Creating seven independent networks
    my_networks = IndependentPerceptrons.IndependentPerceptrons(
        inputs
    )

    return my_networks

def shape_calculation(shape: list, weights: list):
    """
    :param shape: an input that we want to check in the neural network to check its classification
    :param weights: the weights of the neural network
    :return: yes or no answer
    """
    # fix the structure of the input
    for i in range(0, len(shape)):
        shape[i] = file_handling.variable_structuring(shape[i])

    # calculate
    stop = len(shape) - (len(weights) - 1)
    answer = 0

    if stop <= 0:
        for i in range(0, len(shape)):
            answer += shape[i] * weights[i]
    elif stop > 0:
        for i in range(0, len(weights) - 1):
            answer += shape[i] * weights[i]

    answer += weights[len(weights) - 1]

    if answer > 0.2:
        return "this shape is A"

    elif answer < -0.2:
        return "this shape is Not A"

    else:
        return "Uncertain"


def main():
    # creating the neural network
    weights = initialize()
    print("this is the weights after teaching the neural network: ")
    for i in range(0, len(weights) - 1):
        print(weights[i], end="")
        print("(" + "x" + str(i) + ")", end=" + ")
    print(weights[len(weights) - 1])


    my_multi = initialize_multi()
    print("Weights after training Multi-Output Perceptron:")
    letters = ["A", "B", "C", "D", "E", "J", "K"]

    for i in range(7):
        print(f"\nOutput neuron for {letters[i]}:")

        for j in range(63):
            print(f"{my_multi.weight[i][j]}(x{j})", end=" + ")

        print(my_multi.weight_bios[i])

    # Training seven independent perceptrons
    my_independent = initialize_independent()

    letters = ["A", "B", "C", "D", "E", "J", "K"]

    print("\nWeights after training seven independent perceptrons:")

    for letter in range(1, 8):

        network = my_independent.networks[letter]

        print(f"\nNetwork for letter {letters[letter - 1]}:")

        for i in range(63):
            print(
                f"{network.weight[i]}(x{i})",
                end=" + "
            )

        print(network.weight_bios)

main()