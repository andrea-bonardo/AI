import numpy as np
import os
from keras.datasets import mnist
import random
from datetime import datetime, timedelta
from PIL import Image as im
import time
import matplotlib.pyplot as plt
import disegno as d


class NeuronLayer():
    def __init__(self, number_of_outputs, number_of_inputs_per_neuron,number_of_prec_neurons, number_of_layer):
        
        #per creare il file layer.txt e bias.txt da zero
        #check if file exists
        if not os.path.exists("layer.txt"):
            with open("layer.txt", "a") as f:
                for i in range(number_of_inputs_per_neuron):
                    for j in range(number_of_outputs):
                        f.write(f"{(random.randint(51,151)-100)/100} ")
                    f.write("\n")
            f.close()  
        
        if not os.path.exists("bias.txt"):
            with open("bias.txt", "a") as f_bias:
                for i in range(number_of_outputs):
                    f_bias.write(f"{(random.randint(51,151)-100)/100} ")
                    f_bias.write("\n")
            f_bias.close()
        
        
        self.synaptic_weights = np.loadtxt("layer.txt",dtype=float,skiprows=number_of_prec_neurons,max_rows = number_of_inputs_per_neuron)
        self.bias = np.loadtxt("bias.txt",dtype=float,skiprows=number_of_layer,max_rows = number_of_outputs)




class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    def __relu(self, x):
        return np.where(x > 0, x, x * 0.01)

    def __relu_derivative(self, x):
        return np.where(x > 0, 1, -0.01)

    def __softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities
    
    def loss_derivative_respect_softmax(self, softmax, training_set_outputs):
        return softmax - training_set_outputs

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations, learning_rate, n_batch, layer2_graph, lenght, start):
        for self.iteration in range(number_of_training_iterations):

            output_from_layer_1, output_from_layer_2= self.think(training_set_inputs)

            predicted = np.clip(training_set_outputs, 1e-10, 1-1e-10)
            loss = np.sum(-np.log(output_from_layer_2)*predicted)/len(predicted)

            layer2_delta = self.loss_derivative_respect_softmax(output_from_layer_2, training_set_outputs)

            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__relu_derivative(output_from_layer_1)

            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            layer1_biases_adjustment = np.sum(layer1_delta, axis=0)
            layer2_biases_adjustment = np.sum(layer2_delta, axis=0)

            self.layer1.synaptic_weights -= layer1_adjustment*learning_rate
            self.layer1.bias -= layer1_biases_adjustment*learning_rate
            self.layer2.synaptic_weights -= layer2_adjustment*learning_rate
            self.layer2.bias -= layer2_biases_adjustment*learning_rate


            if self.iteration == 0:
                stop=time.time()
                durata=(stop-start)*number_of_training_iterations
                previstoTot = datetime.now() + timedelta(seconds=durata*(lenght-n_batch))
                resultTot = previstoTot.strftime("%I:%M:%S %p")

            if self.iteration > 0 and self.iteration % 100 == 0: 
                learning_rate = learning_rate*0.99
                percentuale=(n_batch+(self.iteration/number_of_training_iterations))/lenght*100            
                os.system("cls")

                print(f"⬜️"*int(percentuale//5),'⬛️'*int((100-percentuale)//5), end=" ")
                print(f"Percentuale: {percentuale:.3f}%")
                print(f"Tempo previsto totale: {resultTot}")

        layer2_graph.append(loss)
        
        with open("layer.txt", "w") as f:
            np.savetxt(f, neural_network.layer1.synaptic_weights)
        with open("layer.txt", "a") as f:
            np.savetxt(f, neural_network.layer2.synaptic_weights)
            f.close()
        with open("bias.txt", "w") as f:
            np.savetxt(f, neural_network.layer1.bias.T)
        with open("bias.txt", "a") as f:
            np.savetxt(f, neural_network.layer2.bias.T)
            f.close()

                
    def think(self, inputs):
        output_from_layer1 = self.__relu(np.dot(inputs, self.layer1.synaptic_weights)+self.layer1.bias)
        output_from_layer2 = self.__softmax(np.dot(output_from_layer1, self.layer2.synaptic_weights)+self.layer2.bias)
        return output_from_layer1, output_from_layer2


if __name__ == "__main__":


    layer1 = NeuronLayer(128,28*28,0,0)
    layer2 = NeuronLayer(10,128,28*28,128)

    neural_network = NeuralNetwork(layer1, layer2)

    [training_set_inputs, training_set_outputs],[x_test, y_test] = mnist.load_data()

    y_n=input("Vuoi fare il training? (y/n) ")

    if y_n == "y":

        training_set_inputs = np.array(training_set_inputs)
        training_set_inputs = training_set_inputs.reshape(60000,28*28)
        training_set_inputs = training_set_inputs/255
        
        training_set_outputs = np.eye(10)[training_set_outputs]

        number_of_training_iterations = 2
        learning_rate = 0.001
        layer4_graph = []

        b_s = 30

        lenght=len(training_set_inputs)//b_s

        layer2_graph = []    

     
        for i in range(lenght):
            start=time.time()
            neural_network.train(training_set_inputs[i*b_s:(i+1)*b_s], training_set_outputs[i*b_s:(i+1)*b_s], number_of_training_iterations, learning_rate, i, layer2_graph, lenght, start)
            neural_network.layer1.synaptic_weights = np.loadtxt("layer.txt",dtype=float,skiprows=0,max_rows = 28*28)
            neural_network.layer1.bias = np.loadtxt("bias.txt",dtype=float,skiprows=0,max_rows = 128)
            neural_network.layer2.synaptic_weights = np.loadtxt("layer.txt",dtype=float,skiprows=28*28,max_rows = 128)
            neural_network.layer2.bias = np.loadtxt("bias.txt",dtype=float,skiprows=128,max_rows = 10)
            os.system("cls")



        print("Percentuale: 100%")
        print("⬜️"*20)
        plt.plot(layer2_graph)
        plt.title('Test')
        plt.xlabel('Iteration')
        plt.ylabel('Errore')
        plt.show(block=False)
        aspetta=input("Premi invio per continuare ")
        plt.close()

    x_test = np.array(x_test)
    x_test = x_test.reshape(10000,28*28)
    x_test = x_test/255

    results = [0,0,0,0,0,0,0,0,0,0]
    times = [0,0,0,0,0,0,0,0,0,0]
    
    hidden_state1,  output = neural_network.think(x_test)
    media=[]
    for i in range(len(x_test)):
        times[y_test[i]]+=1

        if np.argmax(output[i]) == y_test[i]:
            results[y_test[i]]+=1
    for i in range(10):
        media.append(results[i]/times[i])     
        print(f"{i}: {media[i]*100:.3f}%")

    print(f"Totale: {np.mean(media)*100:.3f}%")

    d.DisegnaTu()
    
    image = im.open("image.png").convert('L')
    test=np.asarray(image).reshape(1,28*28)
    test=test/255
    hidden_state1,  output = neural_network.think(test)
    for i in range(10):
        print(f" {output[0][i]*100:.4f}%", end=" ")
    predicted_class = np.argmax(output[0])
    print (f"\n Otteniamo: {predicted_class:.0f} \n")
