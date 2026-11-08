class Perceptron:
    def __init__(self, inputs: list):
        """
        :param inputs: [[[input1], target1] ,[[input2], target2], ...]
        input1 = [element1, ... , element63]
        """

        self.alpha = 1
        self.teta = 0.2
        self.inputs = [] # this a list of lists where each list represents one input
        self.targets = [] # this is a list of target where the target out put of the self.inputs[i] can be found in self.targets[i]
        self.weight = [0] * 63 # weights of the variables
        self.bios = 1
        self.weight_bios = 0
        for i in range(0, len(inputs)):
            # only the inputs that have targets will be considered
            if inputs[i][0] is not None and inputs[i][1] is not None:
                self.inputs.append(inputs[i][0])
                self.targets.append(inputs[i][1])

        self.perceptron()

    def cal_y_in(self, input_: list):

        y_in = self.bios * self.weight_bios

        for i in range(0, len(self.weight)):
            y_in += input_[i] * self.weight[i]

        return y_in

    def cal_y(self, y_in):
        if y_in > self.teta:
            return 1
        elif y_in < (-1 * self.teta):
            return -1
        else:
            return 0

    def epoch(self):
        for i in range(0, len(self.inputs)):
            y_in = self.cal_y_in(self.inputs[i])
            y = self.cal_y(y_in)
            if y != self.targets[i]:
                self.weight_bios += self.bios * self.alpha * self.targets[i]
                for j in range(0, len(self.inputs[i])):
                    if j >= len(self.weight):
                        self.weight.append(0)
                    self.weight[j] += self.alpha * self.inputs[i][j] * self.targets[i]

    def perceptron(self):
        max_epoch = 1000
        for epoch in range(max_epoch):

            old_weights = self.weight.copy()
            old_weight_bios = self.weight_bios

            self.epoch()

            if old_weights == self.weight and old_weight_bios == self.weight_bios:
                    return

        return
