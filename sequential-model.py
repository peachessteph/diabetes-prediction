from numpy import loadtxt #numpy library used to load our dataset
from keras.models import Sequential #neural network with layers linearly stacked, can add layers one at a time
from keras.layers import Dense #fully connected layer

#load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',') #delimiter is one or more characters that separate text strings
#split into input (x) and output (y) variables
x = dataset[:, 0:8] #there are eight input variables and one output variable (the last column)
y = dataset[:, 8]

#1: define keras neural network model, using a fully-connected network (dense) with three layers 
model = Sequential()
#specify the number of nodes in the layer as first argument, and specify the activation function using the activation argument.
#ensure the input layer has the right number of input features with the input_dim argument, 8 for the 8 input variables.
model.add(Dense(12, input_dim=8, activation='relu')) #first layer (input layer)
model.add(Dense(8, activation = 'relu'))
#We will use the rectified linear unit activation function referred to as ReLU on the first two layers and the Sigmoid function in the output layer.
model.add(Dense(1, activation='sigmoid'))

#2: compile the sequential model:
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#Remember training a network means finding the best set of weights to map inputs to outputs in our dataset.
#specify the loss function to use to evaluate a set of weights
#optimizer is used to search through different weights for the network and any optional metrics during training
#use cross entropy as the loss argument. This loss is for a binary classification problems, “binary_crossentropy“
#define the optimizer as the efficient stochastic gradient descent algorithm “adam“

#3: fitting the sequential model
#train or fit model on loaded data (epochs and batches)
model.fit(x, y, epochs=150, batch_size=10)

#4: evaluate the model (on training data)
_, accuracy = model.evaluate(x, y)
print("accuracy: %.2f" % (accuracy*100))

#5: make predictions with model (on validation data)
#using sigmoid activation function on the output layer, so the predictions will be a probability in the range between 0 and 1.
predictions = model.predict(x)
#round predictions
rounded = [round(x[0]) for x in predictions]
#make class predictions with model
predictions = model.predict_classes(x)
#summarize the first 5 cases
for i in range(5):
    print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))
    #as for the percentage sign that's a way to put in variable values within a printed string