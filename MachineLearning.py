from sklearn.cluster import KMeans
from DatasetHandler import *
from matplotlib import pyplot as plt
import Log as log
import time

# the base class for the Machine Learning algorithm
class ML:
    def __init__(self, numClusters):
        self.numClusters = numClusters

    # returns a sum of the areas of all clusters
    def inertia(self):
        pass

    # receives a list of float numbers, and returns the predicted cluster index
    def predict(self, data: list) -> int:
        pass

    # receives a 2D list, that contains all the data that will be used for training the ML
    def fit(self, data: list):
        pass

    def getClustersAmount(self):
        return self.numClusters

# Rom's algorithm
class K_Means(ML):

    def __init__(self, k, tol=0.1, max_iter=300, v=-1):
        super().__init__(k)
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.inertia_values = v

    def fit(self,data1):
        data = np.array(data1)
        self.centroids = {}
#initialization of centroids
        for i in range(self.k):
                self.centroids[i] = data[i]
#optimization process
        for i in range(self.max_iter):
#we are clearing the classifications dictionary for every iteration because for every iteration the centroids move
            self.classifications = {} #contains the centroids and the classifications

            for j in range(self.k):
                    self.classifications[j] = [] #keys=centroids values=feature sets

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids] #creating a list with the distances from every centroid
                classification = distances.index(min(distances)) #choosing the closest centroid's index for the classification
                self.classifications[classification].append(featureset)
#for comparison to check how much the centroids have changed
            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                #finds the mean for all the feature sets(finds better centroids)
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

#checking how much the centroids moved and seeing if their movement is less than the tolerance then their place is not optimized
            optimized = True
            for c in self.centroids:
                    original_centroid = prev_centroids[c]
                    current_centroid = self.centroids[c]
                    if abs(np.linalg.norm(current_centroid-original_centroid)) > self.tol:
                            #print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                            optimized = False
#if optimized equals true it means that the centroid moved very little and we can stop optimizing their placement
            if optimized:
                g =self.inertia()
                #print(self.inertia_values)
                self.inertia_values = g
                #print(self.inertia_values)
                print(i)
                break

    def predict(self,data1): # places the data in their classification by checking the distance to each centroid
        data = np.array(data1)
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

    def inertia(self):
        cluster_sum_of_squares_points_to_clusters = 0
        for centroid, cluster_points in self.centroids.items():
            for cluster_point in cluster_points:
                distance = np.linalg.norm(cluster_point - centroid)
                cluster_sum_of_squares_points_to_clusters += distance**2
        return cluster_sum_of_squares_points_to_clusters

# A simple, temporary implementation of the ML class by using the KMeans algorithm provided by sklearn toolkit
class TempML(ML):
    def __init__(self, numClusters):
        super().__init__(numClusters)
        self.km = KMeans(n_clusters=numClusters)

    def inertia(self):
        return self.km.inertia_

    def predict(self, data: list) -> int:
        return self.km.predict([data])[0]

    def fit(self, data):
        self.km.fit(data)

# A compact version of the MLWrapper, which only includes the predict function and it's prerequisites
class PartialMLWrapper:
    def __init__(self, ml :ML, normalClusters :list, pdatasethandler :PartialDatasetHandler):
        self.ml = ml
        self.normalClusters = normalClusters
        self.dataHandler = pdatasethandler

    def predict(self, data, convert=True):
        datan = data
        if convert:
            datan = self.dataHandler.prepareData(datan)
        return self.ml.predict(datan)

    # a simpler version of the predict function, which returns if a new data is benign or not
    def isNormal(self, data: list, convert=True):
        return self.predict(data, convert) in self.normalClusters

# Provides all the data needed to the algorithm and handles the results- by figuring out the benign clusters, and calculating performance
class MLWrapper:

    # Receives the datahandler of the the train-test datasets, and the number of clusters that are needed to built by the ML
    # if the number of clusters is not provided, the class will automatically try different numbers (from 2 to 40) and will display a graph with the results in the form of the elbow method
    def __init__(self, dataHandler :DatasetHandler, numClusters :int = -1):
        self.dataHandler = dataHandler
        self.ml :ML or None = None
        if numClusters == -1:
           self.__prepareML()
        else:
            self.ml = TempML(numClusters)
        log.show("mlwrapper", "ml prepared")
        self.predictions = []
        self.normalClusters = []
        self.normalAmount = [[0] * self.ml.getClustersAmount(),[0] * self.ml.getClustersAmount()]
        self.attackAmount = [[0] * self.ml.getClustersAmount(),[0] * self.ml.getClustersAmount()]
        self.correctAmount = [0,0] # training 0 , testing 1
        self.FP = [0,0] # training 0 , testing 1
        self.FN = [0,0] # training 0 , testing 1
        self.__setup()

    # the system will try different numbers of clusters (from 2 to 40) and will display a graph with the results in the form of the elbow method
    # NOTE: the program will not close, but rather wait for the input of the requested number of clusters
    def __prepareML(self):
        trainDF = self.dataHandler.getTrainingData()
        allc = []
        for i in range(2, 40):
            ml = K_Means(i)
            ml.fit(trainDF)
            allc.append(ml.inertia()) # needs to be added
            print(f"done: {i}")
        plt.plot(range(2, 40), allc)
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
        self.ml = K_Means(int(input("Number of clusters: ")))

    # trains the ML and calculates the overall performance and row types statistics
    def __setup(self):
        start_time = time.time()
        self.ml.fit(self.dataHandler.getTrainingData())
        log.show("mlwrapper", "fitted")
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        self.__calcStatistics()
        log.show("mlwrapper", "calculated statistics")
        print("--- %s seconds ---" % (time.time() - start_time))
        self.__calcNormal()
        log.show("mlwrapper", "calculated normal groups")
        self.__calcPerformance(testing=False)
        log.show("mlwrapper", "calculated performance of training")

    # calculates the amount of benign rows in each cluster (change testing to True, if you want to calculate the statistics of the testing dataframe)
    def __calcStatistics(self, testing=False):
        loc = 0
        if testing:
            loc = 1
        data = self.dataHandler.getData(testing)
        for i in range(len(data)):
            prediction = self.predict(data[i], convert=testing)
            self.predictions.append(prediction)
            if self.dataHandler.isNormal(i, testing):
                self.normalAmount[loc][prediction] += 1
            else:
                self.attackAmount[loc][prediction] += 1

    # determines what clusters store mostly benign values
    def __calcNormal(self):
        for i in range(self.ml.getClustersAmount()):
            if self.normalAmount[0][i] != 0 and self.attackAmount[0][i] / self.normalAmount[0][i] <= 0.3:
                self.normalClusters.append(i)

    # calculates the overall performance of the ML, such as the amount of correct guesses, and how many of the wrong guesses were false positives (the ML said the data was benign when it wasn't)
    # and how many were false negatives (the ML said the data wasn't benign when it was)
    def __calcPerformance(self, testing=False):
        loc = 0
        if testing:
            loc = 1
        data = self.dataHandler.getData(testing)
        for i in range(len(data)):
            try:
                if not testing:
                    prediction = self.predictions[i] in self.normalClusters
                else:
                    prediction = self.isNormal(data[i], testing)
                fact = self.dataHandler.isNormal(i, testing)
                if (prediction and fact) or (
                        not prediction and not fact):
                    self.correctAmount[loc] += 1
                else:
                    if prediction:
                        self.FP[loc] += 1
                    else:
                        self.FN[loc] += 1
            except IndexError:
                print(f"i: {i}, testing: {testing}")

    # returns a copy of the current class in a PartialMLWrapper format
    def partial(self):
        return PartialMLWrapper(self.ml, self.normalClusters, self.dataHandler.partial())

    # prepares new data to fit the format of the currently fitted dataframe, and return the prediction of it
    # the preparation phase can be skipped by changing convert to False
    def predict(self, data, convert=True):
        datan = data
        if convert:
            datan = self.dataHandler.prepareData(datan)
        return self.ml.predict(datan)

    # a simpler version of the predict function, which returns if a new data is benign or not
    def isNormal(self, data: list, convert=True):
        return self.predict(data, convert) in self.normalClusters

    # tests the ML using the testing dataframe, and calculates the performance
    def test(self):
        self.__calcStatistics(testing=True)
        self.__calcPerformance(testing=True)

    # prints the results of all performance tests that were executed on the ML
    def preview(self):
        print(f"----------- Training: N:{sum(self.normalAmount[0])},A:{sum(self.attackAmount[0])}")
        print(f"Normal ({self.normalClusters}): ", end="")
        for i in range(self.ml.getClustersAmount()):
            print(f"Group {i}: N:{self.normalAmount[0][i]},A:{self.attackAmount[0][i]} | ", end="")
        print("")
        lentrain = len(self.dataHandler.getTrainingData())
        div = (lentrain - self.correctAmount[0])
        if div == 0:
            div = -1
        print(f"Success Rate = {self.correctAmount[0] / lentrain} {self.correctAmount[0]} / {lentrain}")
        print(f"False Positives = {self.FP[0] / div} {self.FP[0]} / {div}")
        print(f"False Negatives = {self.FN[0] / div} {self.FN[0]} / {div}")
        if len(self.dataHandler.getTestingData()) != 0:
            print(f"----------- Testing: N:{sum(self.normalAmount[1])},A:{sum(self.attackAmount[1])}")
            print(f"Groups: ", end="")
            for i in range(self.ml.getClustersAmount()):
                print(f"Group {i}: N:{self.normalAmount[1][i]},A:{self.attackAmount[1][i]} | ", end="")
            print("")
            lentest = len(self.dataHandler.getTestingData())
            div = (lentest - self.correctAmount[1])
            if div == 0:
                div = -1
            print(f"Success Rate = {self.correctAmount[1] / lentest} {self.correctAmount[1]} / {lentest}")
            print(f"False Positives = {self.FP[1] / div} {self.FP[1]} / {div}")
            print(f"False Negatives = {self.FN[1] / div} {self.FN[1]} / {div}")


