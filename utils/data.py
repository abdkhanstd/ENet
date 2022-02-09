"""
Class for managing our data.
"""
import csv
import numpy as np
import random
import glob
import os.path
import sys
import operator
import threading
from utils.processor import process_image
from keras.utils import to_categorical
from tqdm import tqdm   




class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

class DataSet():

    def __init__(self, seq_length, lists=None, class_limit=None, image_shape=(299, 299, 3),features=None,target=None):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.features=features
        self.target=target
        self.lists=lists
        print("Loading list : ",self.lists)
        
        if features==None:
            self.sequence_path = os.path.join('data', 'sequences')
            print('*****Using images only*****')
        else:
            self.sequence_path = os.path.join('data', 'sequences_'+features)
            print('***** Loading sequences from:',self.sequence_path,'*****')

        
        self.max_frames = 2000  # max number of frames a video can have for us to use it

        # Get the data.
        self.data = self.get_data(self)


        # Get the classes.
        self.classes = self.get_classes()
        print('---->>>> Total',target,'Classes:',len(self.classes))


        # Now do some minor data cleaning.
        self.data = self.clean_data()

        self.image_shape = image_shape

    @staticmethod
    def get_data(self):
        """Load our data from file."""
        if self.lists == None:
            to='Splits.csv'
        else:
            to=self.lists
                
            
        with open(os.path.join('data',to), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

        return data


    def clean_data(self):
        """Limit samples to greater than the sequence length and fewer
        than N frames. Also limit it to classes we want to use."""
        data_clean = []
        for item in self.data:
            if int(item[6]) >= self.seq_length and int(item[6]) <= self.max_frames \
                    and item[3] in self.classes and self.target=='views' and not "invalid" in item[3] and not item[3]== '':
                data_clean.append(item)
                
                
            if int(item[6]) >= self.seq_length and int(item[6]) <= self.max_frames \
                    and item[4] in self.classes and self.target=='action' and not "invalid" in item[3] and not item[4]== '':
                data_clean.append(item)    
                
                
            if int(item[6]) >= self.seq_length and int(item[6]) <= self.max_frames \
                    and item[5] in self.classes and self.target=='situation' and not "invalid" in item[3] and not item[5]== '':
                data_clean.append(item)                    

        return data_clean

    def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = []
        if self.target=='views':        
            for item in self.data:
                if item[3] not in classes and not "invalid" in item[3] and not item[3]== '':
                    classes.append(item[3])
                    
        if self.target=='action':        
            for item in self.data:
                if item[4] not in classes and not "invalid" in item[3] and not item[4]== '':
                    classes.append(item[4])

        if self.target=='situation':        
            for item in self.data:
                if item[5] not in classes and not "invalid" in item[3] and not item[5]== '':
                    classes.append(item[5])                    

        # Sort them.
        classes = sorted(classes)
        # Return.
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        #print("################",self.classes.index(class_str))
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))

        assert len(label_hot) == len(self.classes)
        return label_hot
		
    def gen_test(self, train_test, data_type):
            """
            This is a mirror of our generator, but attempts to load everything into
            memory so we can train way faster.
            """
            # Get the right dataset.
            train, test = self.split_train_test()
            data = train if train_test == 'Train' else test
            print("Please wait: Loading %d samples into memory for %sing." % (len(data), train_test))
            
            if train_test == 'Test':
                pbar = tqdm(total=len(data))
                

            X, y = [], []
            z=[]
            
            for row in data:

                if data_type == 'images':
                    frames = self.get_frames_for_sample(row)
                    frames = self.rescale_list(frames, self.seq_length)

                    # Build the image sequence
                    sequence = self.build_image_sequence(frames)

                else:
                    sequence = self.get_extracted_sequence(data_type, row)

                    if sequence is None:
                        print("Can't find sequence. Did you generate them?")
                        raise
                #print ('***********',row[4])
                X.append(sequence)
                y.append(self.get_class_one_hot(row[4]))
                z.append(row[4])
                if train_test == 'Test':
                    pbar.update(1)
            yield np.array(X), np.array(y),z
      

    def split_train_test(self):
        """Split the data into train and test groups."""
        train = []
        test = []
        for item in self.data:
            
            if item[0] == 'Train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    def get_all_sequences_in_memory(self, train_test, data_type):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """

        # Get the right dataset.
        train, test = self.split_train_test()
        data = train if train_test == 'Train' else test
        print("Please wait: Loading %d samples into memory for %sing." % (len(data), train_test))
        pbar = tqdm(total=len(data))

        X, y = [], []
        for row in data:

            if data_type == 'images':
                frames = self.get_frames_for_sample(row)
                frames = self.rescale_list(frames, self.seq_length)
                # Build the image sequence
                sequence = self.build_image_sequence(frames)

            else:
                sequence = self.get_extracted_sequence(data_type, row)

                if sequence is None:
                    print("Can't find sequence. Did you generate them?")
                    raise

            X.append(sequence)
            y.append(self.get_class_one_hot(row[4]))
            pbar.update(1)


        return np.array(X), np.array(y)

    @threadsafe_generator
    def frame_generator(self, batch_size, train_test, data_type):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()

        data = train if train_test == 'Train' else test
        


        print("Creating %s generator with %d samples." % (train_test, len(data)))
        #pbar = tqdm(total=len(data))
        while 1:
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset to be safe.
                sequence = None

                # Get a random sample.
                sample = random.choice(data)

                # Check to see if we've already saved this sequence.
                if data_type == "images":
                    # Get and resample frames.
                    frames = self.get_frames_for_sample(sample)
                    frames = self.rescale_list(frames, self.seq_length)

                    # Build the image sequence
                    sequence = self.build_image_sequence(frames)
                else:
                    # Get the sequence from disk.
                    sequence = self.get_extracted_sequence(data_type, sample)

                    if sequence is None:
                        raise ValueError("Can't find sequence. Did you generate them?")
                        

                X.append(sequence)
                if self.target=='views':
                    y.append(self.get_class_one_hot(sample[3]))
                if self.target=='action':
                    y.append(self.get_class_one_hot(sample[4]))                
                if self.target=='situation':
                    y.append(self.get_class_one_hot(sample[5]))
                    
            yield np.array(X), np.array(y)
            #pbar.update(1)
            

    def build_image_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""
        return [process_image(x, self.image_shape) for x in frames]

    def get_extracted_sequence(self, data_type, sample):
        """Get the saved extracted features."""
        
        tmp=sample[2].split('/')
        tmp=tmp[1].split('.')
        filename = tmp[0]  
        
        path = os.path.join(self.sequence_path, filename + '-' + str(self.seq_length) + \
            '-' + data_type +'_'+self.target+'.npy')
       
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    def get_frames_by_filename(self, filename, data_type):
        """Given a filename for one of our samples, return the data
        the model needs to make predictions."""
        # First, find the sample row.
        sample = None
        for row in self.data:
            if row[2] == filename:
                sample = row
                break
        if sample is None:
            raise ValueError("Couldn't find sample: %s" % filename)

        if data_type == "images":
            # Get and resample frames.
            frames = self.get_frames_for_sample(sample)
            frames = self.rescale_list(frames, self.seq_length)
            # Build the image sequence
            sequence = self.build_image_sequence(frames)
        else:
            # Get the sequence from disk.
            sequence = self.get_extracted_sequence(data_type, sample)

            if sequence is None:
                raise ValueError("Can't find sequence. Did you generate them?")

        return sequence

    @staticmethod
    def get_frames_for_sample(sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        #Getting group        
        tmp=sample[2].split('/')
        tmp=tmp[1].split('.')
        folder_name=tmp[0]
        tmp=tmp[0].split('_')        
        caetagory=sample[1]
        group=tmp[2]
        path = os.path.join('data','SP2',caetagory,group,folder_name)

        images = sorted(glob.glob(os.path.join(path,'*.jpg')))
        return images

    @staticmethod
    def rescale_list(input_list, size):
     

        
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the origina list."""
        #print(len(input_list),'***',size)
        assert len(input_list) >= size

        # Get the number to skip between iterations.
        skip = len(input_list) // size

        # Build our new output.
        output = [input_list[i] for i in range(0, len(input_list), skip)]

        # Cut off the last one if needed.
        return output[:size]