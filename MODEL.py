#IMPORTING REQUIRED MODULES/FUNCTIONS
from keras.utils import to_categorical as tc
from keras.models import Sequential as seq
from keras.layers import Dense, Conv2D, Flatten
#IMPORTING MNIST DATASET
from keras.datasets import mnist

#DATASET SPLITTING
(xtrain,ytrain) , (xtest,ytest) = mnist.load_data()

#PRE-PROCESSING
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
#COMPILING ALL THE LAYERS TOGETHER
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#FITTING THE TRAINING DATASET INTO THE MODEL
model.fit(xtrain, ytrain,epochs=1)

#PREDICTION WITH TESTING DATA
accuracy= (model.evaluate(xtest,ytest))*100
print("Accuracy : ", accuracy[1])

#PUTTING THE ACCURACY DATA INTO A FILE 
try:
	f=open("/accuracy/acc.txt","w")
	f.write(str(int(accuracy[1])))
except:
	print(end="")
finally:
	f.close()
