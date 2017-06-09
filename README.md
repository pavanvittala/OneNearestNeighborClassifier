# OneNearestNeighborClassifier
A supervised learning algorithm written for a Machine Learning course at George Mason.

Implemented a one-nearest neighbor classifier in Java. Using document data as the training set, I developed a classifier
model using the Euclidean distance and cosine similarity measures to find the most similar documents. The model is tested
using leave-one-out cross validation, a method by which each instance of the training data set is tested against the
remaining instances in the data set.

To run the program:

1. Navigate to the folder with the java files.
2. Run the following: javac *.java
3. Run the following: java -Xmx4g Vector data.txt

That's it wait for the code to run and it'll give the output at the end. The dataset is huge so give it time. If you wanna run
it with a smaller dataset then replace data.txt with one of the other txt files in submission (my_half_data.txt, 
my_tiny_data.txt, etc.).
