from mnist_database import mnist_database
from sklearn import cross_validation
from functools import partial
import multiprocessing
import numpy



def test_cl(classifier, ct, tr_data=None, ts_data=None):
    if tr_data is None:
        tr_data = ct.training_data
    if ts_data is None:
        ts_data = ct.testing_data

    guesses = numpy.zeros((10,10), dtype=numpy.int32)

    classifier.fit(tr_data[0][:ct.n_training_vectors],
                   tr_data[1][:ct.n_training_vectors])

    n = 0
    for (t, sol) in zip(ts_data[0],
                        ts_data[1]):
        guesses[sol][classifier.predict(t)] += 1
        n += 1
        if n > ct.n_testing_vectors:
            break

        hit_rate = sum([guesses[d][d] for d in range(len(guesses))]) / float(len(ts_data[0][:ct.n_testing_vectors]))

    return (hit_rate, guesses)


class classifier_test:
    def __init__(self, n_training_vectors=60000, n_testing_vectors=10000):
        # load mnist-database
        self.db = mnist_database('../data/mnist')
        
        self.n_training_vectors = n_training_vectors
        self.n_testing_vectors = n_testing_vectors
        
        self.training_data = self.db.get_training_data()
        self.testing_data = self.db.get_testing_data()
    
    def test_classifier(self, classifier, tr_data=None, ts_data=None):
        return test_cl(self, classifier, tr_data=tr_data, ts_data=ts_data)
    
    def test_classifiers(self, classifiers, processes=1, tr_data=None, ts_data=None):
        pool = multiprocessing.Pool(processes=processes)
        partial_test = partial(test_cl, ct=self, tr_data=tr_data, ts_data=ts_data)
        return pool.map(partial_test, classifiers)
    
    def cross_validate_classifier(self, classifier):
        (images_train, lables_train, images_test, labels_test) = cross_validate.train_test_split(
                self.training_data[0],
                self.training_data[1],
                test_size=0.4, random_state=0)
        
        return self.test_classifier(classifier,
                                    tr_data=(images_train, lables_train),
                                    ts_data=(images_test, labels_test))
    
# workaround for parallelizing instance methods:
# http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

import copy_reg
import types
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
