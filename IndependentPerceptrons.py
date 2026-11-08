import perceptron


class IndependentPerceptrons:

    def __init__(self, inputs: list):

        self.letters = [1, 2, 3, 4, 5, 6, 7]

        self.networks = {}

        # Creating seven independent perceptrons
        for letter in self.letters:

            binary_inputs = []

            # Creating binary targets for the current letter
            for shape, target in inputs:

                if target == letter:
                    binary_target = 1
                else:
                    binary_target = -1

                binary_inputs.append([shape, binary_target])

            # Training one perceptron for this letter
            self.networks[letter] = perceptron.Perceptron(
                binary_inputs
            )

    def predict(self, shape: list):

        if len(shape) != 63:
            raise ValueError(
                "Input must have exactly 63 elements."
            )

        outputs = []

        # Getting the output of all seven networks
        for letter in self.letters:

            network = self.networks[letter]

            y_in = network.cal_y_in(shape)
            y = network.cal_y(y_in)

            outputs.append(y)

        # Checking the classification result
        if outputs.count(1) == 1 and outputs.count(-1) == 6:

            return self.letters[outputs.index(1)]

        return 0