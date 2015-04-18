"""
Raphael Duemig

inspired by:
http://g.sweyla.com/blog/2012/mnist-numpy/
"""

import os, struct
from array import array as c_array
from numpy import append, array, arange, asarray, empty, int8, uint8, zeros

training_images = 'train-images.idx3-ubyte'
training_labels = 'train-labels.idx1-ubyte'
testing_images = 't10k-images.idx3-ubyte'
testing_labels = 't10k-labels.idx1-ubyte'

class mnist_database:
    
    def __init__(self, db_dir='.'):
        self.db_dir = db_dir
    
    # return a tuple containing the array of training images
    # and the array of the corresponding labels
    def get_training_data( self ):
        image_db = self.get_images( os.path.join(self.db_dir, training_images) )
        label_db = self.get_labels( os.path.join(self.db_dir, training_labels) )
        return (image_db, label_db)
    
    def get_testing_data( self ):
        image_db = self.get_images( os.path.join(self.db_dir, testing_images) )
        label_db = self.get_labels( os.path.join(self.db_dir, testing_labels) )
        return (image_db, label_db)
    
    # extract the labels from the database file
    def get_labels( self, db_path ):
        db_file = open( db_path, 'rb' )
        
        # read the db-header
        (magic_nr, size) = struct.unpack('>II', db_file.read(8))
        # read the labels ('b': byte/char)
        labels_raw = c_array('b', db_file.read())
        db_file.close()
        
        # return numpy array
        return asarray(labels_raw, dtype=int8)
    
    # extract the images from the database file
    def get_images( self, db_path ):
        db_file = open( db_path, 'rb' )
        
        # read the db-header
        (magic_nr, size, rows, cols) = struct.unpack('>IIII', db_file.read(16))
        # read the raw image data of all images ('B': unsigned byte/char)
        images_raw = c_array('B', db_file.read())
        db_file.close()

        # the size of an image (pixels)
        img_size = rows*cols
        
        images = empty((size, img_size), dtype=uint8)
        # extract the single images from the byte sequence
        for i in range(size):
            images[i] = array( images_raw[ i*img_size : (i+1)*img_size ] )
        
        return images
