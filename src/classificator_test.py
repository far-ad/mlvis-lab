from mnist_database import mnist_database
from sklearn import cross_validation
import numpy

class classificator_test:
    def __init__(self, n_training_vectors=60000, n_testing_vectors=10000):
        # load mnist-database
        self.db = mnist_database('../data/mnist')
        
        self.n_training_vectors = n_training_vectors
        self.n_testing_vectors = n_testing_vectors
        
        self.training_data = self.db.get_training_data()
        self.testing_data = self.db.get_testing_data()
    
    def test_classifier(self, classifier):
        guesses = numpy.zeros((10,10), dtype=numpy.int32)
        
        classifier.fit(self.training_data[0][:self.n_training_vectors],
                       self.training_data[1][:self.n_training_vectors])
        
        n = 0
        for (t, sol) in zip(self.testing_data[0],
                            self.testing_data[1]):
            guesses[sol][classifier.predict(t)] += 1
            n += 1
            if n > self.n_testing_vectors:
                break
        
        hit_rate = sum([guesses[d][d] for d in range(len(guesses))]) / float(len(self.testing_data[0][:self.n_testing_vectors]))
        
        return (hit_rate, guesses)
    
    def cross_validate_classifier(self, classifier):
        (images_train, lables_train, images_test, labels_test) = cross_validate.train_test_split(
                self.training_data[0],
                self.training_data[1],
                test_size=0.4, random_state=0)
        
        return self.test_classifier(classifier)
    