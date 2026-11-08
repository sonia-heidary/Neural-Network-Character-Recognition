class MultiOutputPerceptron:
    def __init__(self, inputs: list):
        self.alpha = 1
        self.teta = 0.2

        # numbers: A=1, B=2, C=3, D=4, E=5, J=6, K=7
        self.letters = [1, 2, 3, 4, 5, 6, 7]

        self.inputs = []
        self.targets = []


        self.weight = [[0] * 63 for _ in range(7)]
        self.weight_bios = [0] * 7
        self.bios = 1

        for shape, letter in inputs:
            if len(shape) != 63:
                raise ValueError("Each input must have exactly 63 elements.")

            if letter not in self.letters:
                raise ValueError(f"Invalid class number: {letter}")

            self.inputs.append(shape)


            target = [-1] * 7
            target[letter - 1] = 1
            self.targets.append(target)

        self.perceptron()

    def cal_y_in(self, input_, output_index):
        y_in = self.bios * self.weight_bios[output_index]

        for i in range(63):
            y_in += input_[i] * self.weight[output_index][i]

        return y_in

    def cal_y(self, y_in):
        if y_in > self.teta:
            return 1
        elif y_in < -self.teta:
            return -1
        else:
            return 0

    def epoch(self):
        changed = False

        for i in range(len(self.inputs)):
            for k in range(7):
                y_in = self.cal_y_in(self.inputs[i], k)
                y = self.cal_y(y_in)

                if y != self.targets[i][k]:
                    self.weight_bios[k] += (
                        self.alpha * self.targets[i][k] * self.bios
                    )

                    for j in range(63):
                        self.weight[k][j] += (
                            self.alpha
                            * self.inputs[i][j]
                            * self.targets[i][k]
                        )

                    changed = True

        return changed

    def perceptron(self):
        max_epoch = 1000

        for epoch in range(max_epoch):
            changed = self.epoch()

            if not changed:
                print(f"Training converged after {epoch + 1} epochs.")
                return

        print(f"Training stopped after {max_epoch} epochs.")

    def predict(self, shape):
        if len(shape) != 63:
            raise ValueError("Input must have exactly 63 elements.")

        outputs = []

        for k in range(7):
            y_in = self.cal_y_in(shape, k)
            outputs.append(self.cal_y(y_in))


        if outputs.count(1) == 1 and outputs.count(-1) == 6:
            return outputs.index(1) + 1

        return 0
