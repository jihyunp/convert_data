"""
* Original Data Information
  Webpage: http://www.cs.cornell.edu/people/pabo/movie-review-data/
  Downloaded from : http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz

"""

from scipy import io
from scipy.sparse import csc_matrix, lil_matrix, coo_matrix, find
import numpy as np
import os
from datetime import datetime
import gzip
import re
import urllib
import tarfile

class SmallImdb():

    def __init__(self, orig_data_dir='./imdb_2k_orig', output_dir='./imdb_2k', shuffle=True, random_seed=1234):
        '''

        Parameters
        ----------
        orig_data_dir: directory where the dataset is going to be downloaded and unpacked
        output_dir: output directory that will be created
        shuffle: shuffle the indices or not
        random_seed

        '''

        self.orig_data_dir = orig_data_dir
        self.output_dir = output_dir
        self.shuf = shuffle
        self.random_seed = random_seed

        if self.shuf:
            self.shuffled_idx = None

        self.pos_x = None
        self.pos_y = None
        self.neg_x = None
        self.neg_y = None

        self.raw_text = None

        self.docids = None
        self.fold_numbers = None

        self.data_y = None
        self.data_x = None

        self.vocab = None


    def download_and_uncompress(self, url):
        '''
        Called by get_raw_data()

        Parameters
        ----------
        url

        Returns
        -------

        '''
        file_name = url.split('/')[-1]
        file_path = os.path.join(self.orig_data_dir, file_name)

        if not os.path.isfile(file_path):
            print("Downloading " + file_name + " from " + url + " ...")
            urllib.urlretrieve(url, file_path)

        if file_name.split('.')[-1] == 'gz':
            print("Un-compressing data " + file_name)
            if file_name.split('.')[-2] == 'tar':
                tar = tarfile.open(file_path, "r:gz")
                tar.extractall(self.orig_data_dir)
                tar.close()


    def get_raw_data(self):

        """

        Parameters
        ----------
        data_dir : folder path where the data should be downloaded/unpacked
               ex) '/home/datalab/data/imdb_2k

        Returns
        -------

        """
        if not os.path.isdir(self.orig_data_dir):
            os.makedirs(self.orig_data_dir)

        url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz'
        self.download_and_uncompress(url)

        neg_txt_dir = os.path.join(self.orig_data_dir, 'txt_sentoken', 'neg')
        pos_txt_dir = os.path.join(self.orig_data_dir, 'txt_sentoken', 'pos')

        txt_list = []
        label_list = []
        docid_list = []
        fold_num_list = []

        for file_name in os.listdir(neg_txt_dir):
            file_path = os.path.join(neg_txt_dir, file_name)
            cv_num = int(file_name.split('_')[0].split('cv')[-1])
            fold_num = cv_num / 100
            did = int(file_name.split('_')[-1].split('.')[0])
            label = 0
            with open(file_path, 'r') as f:
                txt = f.read()
                txt2 = re.sub('[^a-z0-9\-\' ]', '', txt)
                txt2 = re.sub(' - ', ' ', txt2)
                txt2 = re.sub('\-\-', ' ', txt2)
                txt = re.sub(r'\s+', ' ', txt2)  # removes white spaces
            txt_list.append(txt)
            label_list.append(label)
            docid_list.append(did)
            fold_num_list.append(fold_num)

        for file_name in os.listdir(pos_txt_dir):
            file_path = os.path.join(pos_txt_dir, file_name)
            cv_num = int(file_name.split('_')[0].split('cv')[-1])
            fold_num = cv_num / 100
            did = int(file_name.split('_')[-1].split('.')[0])
            label = 1
            with open(file_path, 'r') as f:
                txt = f.read()
                txt2 = re.sub('[^a-z0-9\-\' ]', '', txt)
                txt = re.sub(' - ', ' ', txt2)
            txt_list.append(txt)
            label_list.append(label)
            docid_list.append(did)
            fold_num_list.append(fold_num)

        self.raw_text = txt_list
        self.docids = np.array(docid_list)
        self.fold_numbers = np.array(fold_num_list)
        self.data_y = np.array(label_list, dtype=np.int16)



    def get_matrices(self):

        print('Generating BOW matrix')
        from sklearn.feature_extraction.text import CountVectorizer
        from nltk.corpus import stopwords

        raw_txt = self.raw_text

        count_vec = CountVectorizer(stop_words=stopwords.words('english'))
        bow = count_vec.fit_transform(raw_txt)
        vocab = count_vec.vocabulary_
        vocab_inv = {y:x for x,y in vocab.iteritems()}

        # Change vocab order
        sorted_vocab_idx = np.argsort(np.array(bow.sum(axis=0))[0])[::-1]
        new_bow = bow[:,sorted_vocab_idx]
        new_vocab = [vocab_inv[i] for i in sorted_vocab_idx]

        if self.shuf:
            import random
            shuffled_idx = range(new_bow.shape[0])
            random.seed(self.random_seed)
            random.shuffle(shuffled_idx)
            new_bow = new_bow[shuffled_idx,:]
            self.data_y = self.data_y[shuffled_idx]
            self.docids = self.docids[shuffled_idx]
            self.fold_numbers= self.fold_numbers[shuffled_idx]
            self.shuffled_idx = shuffled_idx
            self.raw_text = [self.raw_text[i] for i in shuffled_idx]

        self.data_x = new_bow
        self.vocab = new_vocab


    def print_svmlight_format(self, x_data, y_data, bow_txt_file):

        print('Printing BOW ..')
        f = open(bow_txt_file, 'w')
        for i, catid in enumerate(y_data):
            if catid == 0:
                lab = '-1'
            else:
                lab = '1'
            f.write('{}'.format(lab))
            bow = find(x_data[i, :])
            for num, wid in zip(bow[2], bow[1]):
                num = int(num)
                f.write(' {}:{}'.format(wid, num))
            f.write('\n')
        f.close()


    def print_vocab(self, output_file='imdb_2k.vocab'):

        print('Printing vocabulary ')
        # Sort in descending order
        output_vocab_file = open(output_file, 'w')
        for word in self.vocab:
            output_vocab_file.write(word + '\n')
        output_vocab_file.close()


    def print_ids_foldnums(self):
        print('Printing review ids and fold numbers')
        review_ids_file = os.path.join(self.output_dir, 'imdb_2k.docids')
        f = open(review_ids_file, 'w')
        for id in self.docids:
            f.write(str(id) + '\n')
        f.close()
        fold_num_file = os.path.join(self.output_dir, 'imdb_2k.cvtag')
        f2 = open(fold_num_file, 'w')
        for id in self.fold_numbers:
            f2.write(str(id) + '\n')
        f2.close()



    def save_and_print_data(self):

        # from copy import copy
        import cPickle as cp

        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        bow_file = os.path.join(self.output_dir, 'labeledBow.feat')
        cp_file = os.path.join(self.output_dir, 'dataset.pkl')

        dataset = (self.data_x, self.data_y)
        print('Saving X matrix and label array to dataset.pkl ..')
        cp.dump(dataset, open(cp_file,'wb'), protocol=cp.HIGHEST_PROTOCOL)

        self.print_svmlight_format(self.data_x, self.data_y, bow_file)

        # Printing vocabulary
        self.print_vocab(self.output_dir + '/imdb_2k.vocab')



    def print_text_file(self, output_folder='./imdb_2k'):
        '''
        This is for running https://github.com/hiyijian/doc2vec

        Parameters
        ----------
        output_folder

        Returns
        -------

        '''

        print('Printing text file in doc2v')
        # Now printing text
        text_output_file = os.path.join(output_folder, 'text.txt')
        outfile = open(text_output_file, 'w')
        for did, text in zip(self.docids, self.raw_text):
            line = '_*' + str(did) + ' ' + text + '\n'
            outfile.write(line.encode("utf-8"))
        outfile.close()





if __name__ == "__main__":


    obj = SmallImdb(orig_data_dir='./imdb_2k_orig', output_dir='./imdb_2k', )
    obj.get_raw_data()
    obj.get_matrices()
    obj.save_and_print_data()
    obj.print_text_file()
    obj.print_ids_foldnums()








