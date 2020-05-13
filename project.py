import numpy as np


class errors:

    @staticmethod
    def mse(real, predicted):
        return np.sqrt(np.square(np.subtract(real, predicted)).mean())


class activations:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.e ** -x)

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)


class network:
    def __init__(self, inputs, targets, hidden_layer_neurons=2, learning_rate=0.5):
        self.inputs = inputs
        self.targets = targets
        self.hidden_layer_neurons = hidden_layer_neurons
        self.learning_rate = learning_rate
        self.weights_layer_1 = np.random.uniform(
            low=-0.5, high=0.6, size=(inputs.shape[1], hidden_layer_neurons + 1))
        self.weights_layer_2 = np.random.uniform(
            low=-0.5, high=0.6, size=(hidden_layer_neurons + 1, 1))
        self.activations = activations
        self.errors = errors

    def fit(self, epochs):
        for i in range(epochs):
            # 1a pesos entre input and hidden
            self.v_i = np.dot(self.inputs, self.weights_layer_1)
            # 1b activacion sigmoid de 1a
            self.output_v_i = self.activations.sigmoid(self.v_i)
            # 1c
            self.w_i = np.dot(self.output_v_i, self.weights_layer_2)
            # 1d
            self.output_w_i = self.activations.sigmoid(self.w_i)

            # 2 Backpropagation
            # a. Output Layer
            # i
            self.local_error = np.subtract(self.targets, self.output_w_i)
            # ii
            # 1
            self.predicted_derivate1 = self.activations.sigmoid_derivative(
                self.output_w_i)
            # 2 deltas
            self.deltas1 = np.multiply(
                self.predicted_derivate1, self.local_error)

            # b. Hidden Layer
            # i
            self.new_error = np.dot(
                self.deltas1, np.transpose(self.weights_layer_2))
            # ii deltas
            # 1
            self.predicted_derivate2 = self.activations.sigmoid_derivative(
                self.output_v_i)
            # 2 deltas
            self.deltas2 = np.multiply(
                self.new_error, self.predicted_derivate2)

            # c. Adjusts weights
            # i
            self.aw_i = np.dot(np.transpose(self.output_v_i), self.deltas1)
            # ii
            self.weights_layer_2 += np.multiply(self.aw_i, self.learning_rate)
            # iii
            self.av_i = np.dot(np.transpose(self.inputs), self.deltas2)
            # iv
            self.weights_layer_1 += np.multiply(self.av_i, self.learning_rate)

    def predict(self, inputs):
        # i
        self.pv_i = np.dot(inputs, self.weights_layer_1)
        # ii
        self.z_i = self.activations.sigmoid(self.pv_i)
        # iii
        self.pw_i = np.dot(self.z_i, self.weights_layer_2)
        # iv
        self.y_i = self.activations.sigmoid(self.pw_i)
        # v
        return self.y_i


class Measures():
    def __init__(self, real, pred):
        self.real = real
        self.pred = pred

        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0

        for i in range(len(self.real)):
            if self.real[i] <= 0 and self.pred[i] <= 0:
                self.tn += 1
            elif self.real[i] <= 0 and self.pred[i] > 0:
                self.fp += 1
            elif self.real[i] > 0 and self.pred[i] <= 0:
                self.fn += 1
            elif self.real[i] > 0 and self.pred[i] > 0:
                self.tp += 1

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def f1(self):
        p = self.precision()
        r = self.recall()
        return 2 * (p * r) / (p + r)

    def accuracy(self):
        return (self.tp + self.tn) / (self.tn + self.fp + self.fn + self.tp)

    def cmatrix(self):
        cm = [
            [self.tp, self.fp],
            [self.fn, self.tn]
        ]

        return cm


def train(inputs, targets):
    net = network(inputs, targets)
    epochs = 10000
    net.fit(epochs)
    return net


if __name__ == "__main__":

    for i in range(1, 7):
        file_object = open("data"+str(i)+".txt", "r")
        f1 = file_object.readlines()
        inputs = []
        targets = []
        for x in f1[1:]:
            data = list(map(int, x.strip().split()))
            inputs.append([inp for inp in data[:3]])
            targets.append([data[3]])

        inputs = np.array(inputs)
        targets = np.array(targets)

        net = train(inputs, targets)

        predicted = net.predict(inputs)
        m = Measures(targets, predicted)

        print("----------data"+str(i)+"-------------")
        print("----Measures----")

        print("precision %s" % m.precision())
        print("recall %s" % m.recall())
        print("f1 %s" % m.f1())
        print("accuracy %s" % m.accuracy())

        track = (np.abs(predicted - targets) <= 0.2).all(axis=1)
        print()
        print("----Result----")
        try:
            assert(np.array_equal(track, np.array([True, True, True, True])))
            print("PREDICT PASSED!")
        except:
            print("PREDICT NOT PASSED!")
        print("\n\n")
        #print("pred:", predicted)
        #print("targ:", targets)
