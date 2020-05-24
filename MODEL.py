#IMPORTING MNIST DATASET
from keras.datasets import mnist
from keras.utils import to_categorical as tc
from keras.models import Sequential as seq
from keras.layers import Dense, Conv2D, Flatten

#DATASET SPLITTING
(xtrain,ytrain) , (xtest,ytest) = mnist.load_data()

#RESHAPING
xtrain=xtrain.reshape(60000,28,28,1)
xtest=xtest.reshape(10000,28,28,1)

#CONVERTING OUTPUT TO CATEGORICAL DATA
ytrain=tc(ytrain)
ytest=tc(ytest)

#CREATING MODEL
model = seq()
model.add(Conv2D(2, kernel_size=3, activation="relu", input_shape=(28,28,1)))

#HYPERPARAMETERS : FilterSize, NbrOfConvLayers
NbrOfConvLayers=1
FilterSize=4
for NbrOfConvLayers in range(NbrOfConvLayers):
	model.add(Conv2D(filters=FilterSize, kernel_size=3, activation="relu"))
	FilterSize*=2

model.add(Flatten())
model.add(Dense(10, activation="softmax"))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(xtrain, ytrain,epochs=1)

#PREDICTION
pred= (model.evaluate(xtest,ytest))*100
print("Accuracy : ", pred[1])

#PUTTING THE ACCURACY DATA INTO A FILE 
try:
	f=open("/accuracy/acc.txt","w")
	f.write(str(int(pred[1])))
except:
	print(end="")
finally:
	f.close()
