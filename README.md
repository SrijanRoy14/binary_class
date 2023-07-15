# binary_class
A binary classification of image using cnn.
the data used here is a 175 no. lf images divided equally into two classes: happy and sad. The data is basic google images.
keras.data is used to load the datapipeline.
the model is a sequential model with conv2d and maxpooling layers as the filter layers and 2 dense layers has been used to return a single o/p. No regularization has been used.
the model is stored as an api in the models section.
also callbacks are set in the logs so as to create a checkpoint and data visualization.
tensorflow==2.80 and tensorflow-gpu==2.80 has been used

