import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
    
    
    # first center chosen at random
    N,D = x.shape
    rand_center = generator.randint(0,n,1)
    rand_center =0
    c1 = x[rand_center,:]
    if len(c1.shape) != 2:
        c1 = c1[np.newaxis,:]
        #print(c1.shape)
    centroids = np.array([]).reshape(0,D)
    
    #centroids = []
    #centroids.append([])
    #centroids.append(c1)
    #print(centroids)
    centroids = np.append(centroids,c1,axis = 0)
    centroidsIndex = []
    centroidsIndex.append(rand_center)
    distance = []
    #centroids = np.asarray(centroids)
    #centroids = centroids[:,np.newaxis]
    
    for j in range(n_cluster - 1):
        distance = []
        for i in range(n):
            #print(centroids.shape)
            cluster_euclid = np.sum((centroids - x[i])**2,axis = 1)
            min_distance = np.amin(cluster_euclid)
            distance.append(min_distance)
        #print(distance)
        max_index = np.argmax(distance)
        temp = x[max_index,:]
        temp = temp[np.newaxis,:]
        centroids = np.append(centroids,temp,axis = 0)
        centroidsIndex.append(max_index)
    #print(centroidsIndex)
    centers = centroidsIndex
    """for j in range(n_cluster -1):
        for i in range(len(centroids)):
            distance.append([])
            distanceList = np.subtract(x,centroids[i])
            distanceList = np.power(x,2)
            distance.append(distanceList)"""    
            
    #raise Exception(
    #         'Implement get_k_means_plus_plus_center_indices function in Kmeans.py')

    

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers



def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        
        #centroidsIndex = self.generator.choice(N,self.n_cluster, replace = False)
        centroidsIndex = self.centers
        centroids = x[centroidsIndex,:]
        y = np.zeros(N,dtype = int)
        
        its = 0
        #self.max_iter = 2
        distance = np.array([]).reshape(N,0)
        
        while its < self.max_iter:
            #computer distance from each point to each centroid
            for k in range(self.n_cluster):
                euclid = np.sum((x-centroids[k,:])**2,axis=1)
                euclid = euclid[:,np.newaxis]
                distance = np.append(distance,euclid, axis = 1)
            
            y_temp = np.argmin(distance, axis =1)
            #reset distance array fro next iterations
            distance = np.array([]).reshape(N,0)
            
            #compute new means
            new_centroids = np.array([np.mean(x[y_temp == k],axis =0) for k in range(self.n_cluster)])
            #handle empty clusters
            empty = np.where(np.isnan(new_centroids))
            new_centroids[empty]=centroids[empty]
            
            #compute convergence
            converged = True
            for k in range(self.n_cluster):
                o_centroid = centroids[k]
                n_centroid = new_centroids[k]
                if np.sum(np.absolute(o_centroid - n_centroid)) > self.e:
                    converged = False
            if converged == True:
                break
            centroids = new_centroids
            its = its+1
        y = y_temp
        self.max_iter = its
        
        #raise Exception(
        #     'Implement fit function in KMeans class')
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, self.max_iter

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
        #     'Implement fit function in KMeansClassifier class')

        kmeans = KMeans(self.n_cluster,self.max_iter,self.e,self.generator)
        centroids, y_cluster, max_iters = kmeans.fit(x)
        labels = []
        
        #ToDo: assign smaller label in case of similar maximums
        
        for k in range(self.n_cluster):
            sample = {}
            cluster_array = y[y_cluster == k]
            #print(cluster_array)
            if len(cluster_array) == 0:
                labels.append(0)
                continue
            for i in range(len(cluster_array)):
                if sample.get(cluster_array[i]) == None:
                    sample[cluster_array[i]] = 1
                else:
                    sample[cluster_array[i]] = sample[cluster_array[i]] +1
            max_label = max(sample, key = sample.get)
            #print(max_label)
            labels.append(max_label)            
        centroid_labels = np.array(labels)
        
        
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
        #     'Implement predict function in KMeansClassifier class')
        
        distance = np.array([]).reshape(N,0)
        for k in range(self.n_cluster):
            euclid = np.sum((x-self.centroids[k,:])**2,axis=1)
            euclid = euclid[:,np.newaxis]
            distance = np.append(distance,euclid, axis = 1)    
        y_temp = np.argmin(distance, axis =1)
        y_label = self.centroid_labels[y_temp]
        #print(y_label)
        
        labels = y_label
        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    #raise Exception(
    #         'Implement transform_image function')
    
    #reshape image:
    x,y,z = image.shape
    img_2d = image.reshape(-1,z)
    
    #kmeans = KMeans(self.n_cluster,self.max_iter,self.e,self.generator)
    #centroids, y_cluster, max_iters = kmeans.fit(img_2d)
    
    no_cluster,d = code_vectors.shape
    distance = np.array([]).reshape(x*y,0)
        
    for k in range(no_cluster):
        euclid = np.sum((img_2d-code_vectors[k,:])**2,axis=1)
        euclid = euclid[:,np.newaxis]
        if k==0:
            temp =euclid
        distance = np.append(distance,euclid, axis = 1)    
    y_temp = np.argmin(distance, axis =1)
    
    new_img = np.empty([x*y,z])
    for i in range(x*y):
        new_img[i] = code_vectors[y_temp[i]]
    new_im = new_img.reshape(x,y,z)
    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im

