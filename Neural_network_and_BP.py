import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    y=sigmoid(x)
    return y(1-y)

class Network():
    def __init__(self, num_layers,num_neurons_each_layer,first_neurons_input):
        # input:
        # num_layers = int
        # num_neurons_each_layer = [] each num represents number of neurons in each layer respectively
        # first_neurons_input
        self.neurons=[] # each an np.array represent the values of the neuron
        self.weights=[] # each an np.array represent the weight values
        self.bias=np.ones(num_layers) #bias
        self.outputs=np.zeros(num_neurons_each_layer[-1]) # an np.array of the neuron number in last layer 
        self.input_neurons=first_neurons_input

        for _num_layer in range(num_layers):
            self.neurons.append(np.zeros(num_neurons_each_layer[_num_layer]))
            self.weights.append(np.zeros((num_neurons_each_layer[_num_layer-1]+1 if _num_layer>0 else len(first_neurons_input)+1,num_neurons_each_layer[_num_layer])))


    def predict(self, first_neurons_input,act_func):
        for _ in range(len(self.neurons)):
            if _==0:
                self.neurons[_]=act_func(np.dot(np.append(self.input_neurons,np.array((1))),self.weights[_]))
            else:
                self.neurons[_]=act_func(np.dot(np.append(self.neurons[_-1],np.array((1))),self.weights[_]))
        return self.neurons[-1]

    def layout(self):
        print('# of Neuron for layer {} \t: {}\n{}'.format(0,len(self.input_neurons),self.input_neurons))

        for _ in range(len(self.neurons)):
            print('Weight Matrix #{}:\n{}'.format(_+1,self.weights[_]))
            print('# of Neuron for layer {} \t: {}\n{}'.format(_+1,len(self.neurons[_]),self.neurons[_]))



if __name__=='__main__':
    network=Network(4,[6,5,6,3],[2,3,4])
    network.layout()
    print(network.predict([2,3,4],sigmoid))


print()