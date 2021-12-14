import numpy as np



def sigmoid(s, deriv=False):
    if deriv:
        return s * (1 - s)
    return 1/(1 + np.exp(-s))



class NeuralNetwork:
    def __init__(self):
        #parameters
        self.inputSize = 784
        self.outputSize = 10
        self.hidden_layers = [10]

        #weights
        self.weights = []
        self.weights.append(np.random.randn(self.inputSize, self.hidden_layers[0])) # in -> hidden
        for i in range(len(self.hidden_layers)-1):
            self.weights.append(np.random.randn(self.hidden_layers[i],self.hidden_layers[i+1]))
        self.weights.append(np.random.randn(self.hidden_layers[-1], self.outputSize)) # hidden -> out
        self.neurons = []
        #print(self.weights[2])
    def feedForward(self, X):
        #forward propogation through the network
        current = X.copy()
        self.neurons.clear()
        #print(len(self.weights),'l')
        for w in self.weights:
            #print(current.shape,w.shape)
            z = np.dot(current,w)
            self.neurons.append(current)
            current = sigmoid(z)

            #print(z.shape,current.shape)
        #self.z = np.dot(X, self.weights[0]) #dot product of X (input) and first set of weights (3x2)
        #self.z2 = sigmoid(self.z) #activation function
        #print(self.z.shape,self.z2.shape)
        #self.z3 = np.dot(self.z2, self.W2) #dot product of hidden layer (z2) and second set of weights (3x1)
        return current

    def backward(self, X, y, output):
        #backward propogate through the network
        self.output_error = y - output # error in output
        self.output_delta = self.output_error * sigmoid(output, deriv=True)
        current_delta = self.output_delta

        for i in range(len(self.neurons)-1,0,-1):
            # print(current_delta.shape,self.weights[i].T.shape,self.neurons[i].T.shape)
            err = current_delta.dot(self.weights[i].T)
            x=np.dot(self.neurons[i].T,current_delta)
            self.weights[i] += x #.dot()
            current_delta = err * sigmoid(self.neurons[i])

    def train(self, X, y):
        output = self.feedForward(X)
        self.backward(X, y, output)


# X = (hours sleeping, hours studying), y = test score of the student
#X = np.array(([2, 9], [1, 5], [3, 6],[1,1]), dtype=float)
#y = np.array(([92], [86], [89],[50]), dtype=float)


# scale units
#X = X/np.amax(X, axis=0) #maximum of X array
#y = y/100 # maximum test score is 100

# ['training_images', 'training_labels', 'test_images', 'test_labels', 'validation_images', 'validation_labels']
with np.load('mnist.npz') as f:
    training_images = f['training_images'].reshape((50000,784))
    training_labels = f['training_labels'].reshape((50000,10))

    test_images = f['test_images']
    test_labels = f['test_labels']

NN = NeuralNetwork()

for i in range(1000): #trains the NN 1000 times
    if i % 10 == 0: print(i)
    if i % 20 == 0:
        print("Loss: " + str(np.mean(np.square(training_labels - NN.feedForward(training_images)))))
    NN.train(training_images, training_labels)

print("Input: " + str(training_images))
print("Actual Output: " + str(training_labels))
print("Loss: " + str(np.mean(np.square(training_labels - NN.feedForward(training_images)))))
print()
out = NN.feedForward(training_images)
print("Predicted Output: " + str(out))
print('\n\n')
print(training_labels[0],'\n')
print(out[0])