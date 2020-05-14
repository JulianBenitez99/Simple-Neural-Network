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
        self.weights_layer_1 = np.random.random(
            size=(inputs.shape[1], hidden_layer_neurons + 1))
        self.weights_layer_2 = np.random.random(
            size=(hidden_layer_neurons + 1, 1))
        self.activations = activations
        self.errors = errors

    def fit(self, epochs):
        for i in range(epochs):
            # 1a pesos entre input and hidden
            v_i = np.dot(self.inputs, self.weights_layer_1)
            # 1b activacion sigmoid de 1a
            output_v_i = self.activations.sigmoid(v_i)
            # 1c
            w_i = np.dot(output_v_i, self.weights_layer_2)
            # 1d
            output_w_i = self.activations.sigmoid(w_i)

            # 2 Backpropagation
            # a. Output Layer
            # i
            local_error = np.subtract(self.targets, output_w_i)
            # ii
            # 1
            predicted_derivate1 = self.activations.sigmoid_derivative(
                output_w_i)
            # 2 deltas
            deltas1 = np.multiply(
                predicted_derivate1, local_error)

            # b. Hidden Layer
            # i
            new_error = np.dot(
                deltas1, np.transpose(self.weights_layer_2))
            # ii deltas
            # 1
            predicted_derivate2 = self.activations.sigmoid_derivative(
                output_v_i)
            # 2 deltas
            deltas2 = np.multiply(
                new_error, predicted_derivate2)

            # c. Adjusts weights
            # i
            aw_i = np.dot(np.transpose(output_v_i), deltas1)
            # ii
            self.weights_layer_2 += np.multiply(aw_i, self.learning_rate)
            # iii
            av_i = np.dot(np.transpose(self.inputs), deltas2)
            # iv
            self.weights_layer_1 += np.multiply(av_i, self.learning_rate)

    def predict(self, inputs):
        # i
        pv_i = np.dot(inputs, self.weights_layer_1)
        # ii
        z_i = self.activations.sigmoid(pv_i)
        # iii
        pw_i = np.dot(z_i, self.weights_layer_2)
        # iv
        y_i = self.activations.sigmoid(pw_i)
        # v
        return y_i


class Measures:

    def __init__(self, real, pred):
        self.real = real
        self.pred = pred

        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0

        for i in range(len(self.real)):
            if self.real[i] <= 0.5 and self.pred[i] <= 0.5:
                self.tn += 1
            elif self.real[i] <= 0.5 < self.pred[i]:
                self.fp += 1
            elif self.real[i] > 0.5 >= self.pred[i]:
                self.fn += 1
            elif self.real[i] > 0.5 and self.pred[i] > 0.5:
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


def get_trained_net(inputs, targets):
    net = network(inputs, targets)
    epochs = 10000
    net.fit(epochs)
    return net


if __name__ == "__main__":

    for i in range(1, 7):
        file_object = open("data" + str(i) + ".txt", "r")
        data_file = file_object.readlines()
        inputs = []
        targets = []
        n = int(data_file[0])
        for j in range(1, n + 1):
            line = data_file[j]
            data = list(map(int, line.strip().split()))
            inputs.append([inp for inp in data[:len(line) - 1]])
            targets.append([data[-1]])
        inputs = np.array(inputs)
        targets = np.array(targets)

        net = get_trained_net(inputs, targets)

        predicted = net.predict(inputs)
        m = Measures(targets, predicted)

        print("----------data" + str(i) + "-------------")
        print("----Measures----")

        print("precision %s" % m.precision())
        print("recall %s" % m.recall())
        print("f1 %s" % m.f1())
        print("accuracy %s" % m.accuracy())

        track = (np.abs(predicted - targets) <= 0.2).all(axis=1)
        print()
        print("----Result----")
        print("predicted:\n", predicted)
        print("target:\n", targets)
        try:
            assert (np.array_equal(track, np.array([True, True, True, True])))
            print("PREDICT PASSED!")
        except:
            print("PREDICT NOT PASSED!")
        print("\n")
