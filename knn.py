import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function

    # TODO: save features and lable to self
    def train(self, features, labels):
        """
        In this function, features is simply training data which is a 2D list with float values.
        For example, if the data looks like the following: Student 1 with features age 25, grade 3.8 and labeled as 0,
        Student 2 with features age 22, grade 3.0 and labeled as 1, then the feature data would be
        [ [25.0, 3.8], [22.0,3.0] ] and the corresponding label would be [0,1]

        For KNN, the training process is just loading of training data. Thus, all you need to do in this function
        is create some local variable in KNN class to store this data so you can use the data in later process.
        :param features: List[List[float]]
        :param labels: List[int]
        """
        self.train_label = labels
        self.features = features
        #raise NotImplementedError

    # TODO: predict labels of a list of points
    def predict(self, features):
        """
        This function takes 2D list of test data points, similar to those from train function. Here, you need process
        every test data point, reuse the get_k_neighbours function to find the nearest k neighbours for each test
        data point, find the majority of labels for these neighbours as the predict label for that testing data point.
        Thus, you will get N predicted label for N test data point.
        This function need to return a list of predicted labels for all test data points.
        :param features: List[List[float]]
        :return: List[int]
        """

        predicted_label = []
        # features = [[20, 1], [3, 4]]
        for i in features:
            labels = self.get_k_neighbors(i)
            print("label returned by KNN: ", labels)
            # find majority of labels:
            count = Counter(labels)
            print(count.most_common(1)[0][0])
            predicted_label.append(count.most_common(1)[0][0])
            print("PredictedLabel", predicted_label)
        return predicted_label
        #raise NotImplementedError

    # TODO: find KNN of one point
    def get_k_neighbors(self, point):
        """
        This function takes one single data point and finds k-nearest neighbours in the training set.
        You already have your k value, distance function and you just stored all training data in KNN class with the
        train function. This function needs to return a list of labels of all k neighours.
        :param point: List[float]
        :return:  List[int]
        """
        dist = []
        features = [[0, 10], [2, 0], [0, 0]]
        labels = [0, 1, 0]
        #features = self.features
        #labels = self.train_label
        for i in range(len(features)):
            #value = self.distance_function(point, self.features[i])
            distance = 0
            for p, q in zip(features[i], point):
                distance += (p - q) ** 2
            dist.append(np.sqrt(distance))
        j = 0
        # indices = []
        k_labels = []
        dist = np.asarray(dist)
        for k in range(2):
            j = np.argmin(dist)
            k_labels.append(labels[j])
            print("Dist array: ", dist)
            # print("Distance index: ", np.delete(dist, dist.argmin(dist)))
            #dist = np.delete(dist, dist.index(min(dist)))
            dist = np.delete(dist, np.argmin(dist))
            labels = np.delete(labels, np.argmin(labels))
            print("Dist Array after deletion ", dist)
            print(" k_labels: ", k_labels)
        return k_labels
        # raise NotImplementedError


if __name__ == '__main__':
    # print(np.__version__)
    a = KNN(2, 'euclidean')
    a.predict()

