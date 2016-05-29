

from scipy import io
from scipy.sparse import csc_matrix, lil_matrix, coo_matrix, find
import numpy as np
import os
from datetime import datetime
import gzip
import re
import urllib
import tarfile
from process_imdb_2k import SmallImdbData


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from ProcessData import ProcessData
from ProcessData import _split_data


class NewsGroupsData(ProcessData):


    def __init__(self, output_dir, categories, train_valid_split=(8, 2), sup_unsup_split=(10, 300),
                 train_test_split=(8, 2), shuffle=False, random_seed=1234):
        '''

        Parameters
        ----------
        orig_data_dir
        output_dir
        train_valid_split
        sup_unsup_split
        train_test_split
        shuffle
        random_seed
        '''

        ProcessData.__init__(self, '', output_dir, train_valid_split, sup_unsup_split, train_test_split, shuffle, random_seed)

        self.categories = categories
        self.all_categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                               'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
                               'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
                               'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
                               'talk.politics.misc', 'talk.religion.misc']

        self.binary=False
        if len(categories) == 2:
            self.binary=True



    def get_raw_data(self):

        categories = self.categories
        train_20ng = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories)
        test_20ng = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories)

        txt_list = []
        y_lab = []
        self.train_docids = []

        for j, line in enumerate(train_20ng.data):
            line = line.lower()
            line = re.sub('[^a-z0-9\-\' ]', ' ', line)
            line = re.sub(' ---*', ' ', line)
            line = re.sub(r'\s+', ' ', line)
            if len(line) != 0:
                txt_list.append(line)
                y_lab.append(train_20ng.target[j])
                self.train_docids.append(j)

        self.train_raw_text = txt_list
        self.train_y = np.array(y_lab)

        txt_list = []
        y_lab = []
        self.test_docids = []

        train_len = len(train_20ng.target)
        test_len = len(test_20ng.target)

        for i, line in enumerate(test_20ng.data):
            line = line.lower()
            line = re.sub('[^a-z0-9\-\' ]', ' ', line)
            line = re.sub(' ---*', ' ', line)
            line = re.sub(r'\s+', ' ', line)
            if len(line) != 0:
                txt_list.append(line)
                y_lab.append(test_20ng.target[i])
                self.test_docids.append(i + train_len)

        self.test_raw_text = txt_list
        self.test_y = np.array(y_lab)

        # For unsup data

        unsup_categories = [cat for cat in self.all_categories if cat not in self.categories]
        unsup_20ng = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), categories=unsup_categories)

        txt_list = []
        self.unsup_docids = []

        for k, line in enumerate(unsup_20ng.data):
            line = line.lower()
            line = re.sub('[^a-z0-9\-\' ]', ' ', line)
            line = re.sub(' ---*', ' ', line)
            line = re.sub(r'\s+', ' ', line)
            if len(line) != 0:
                txt_list.append(line)
                self.unsup_docids.append(train_len + test_len + k)

        self.unsup_raw_text = txt_list




    def get_matrices(self):
        print('Generating BOW matrix')
        from sklearn.feature_extraction.text import CountVectorizer
        from nltk.corpus import stopwords

        count_vec = CountVectorizer(stop_words=stopwords.words('english'))
        count_vec = count_vec.fit(self.train_raw_text + self.unsup_raw_text)
        bow = count_vec.transform(self.train_raw_text)
        bow_test = count_vec.transform(self.test_raw_text)
        bow_unsup = count_vec.transform(self.unsup_raw_text)
        vocab = count_vec.vocabulary_
        vocab_inv = {y: x for x, y in vocab.iteritems()}

        # Change vocab order
        sorted_vocab_idx = np.argsort(np.array(bow.sum(axis=0))[0])[::-1]
        new_bow = bow[:, sorted_vocab_idx]
        new_bow_test = bow_test[:, sorted_vocab_idx]
        new_bow_unsup = bow_unsup[:, sorted_vocab_idx]
        new_vocab = [vocab_inv[i] for i in sorted_vocab_idx]

        self.train_x = new_bow
        self.test_x = new_bow_test
        self.unsup_x = new_bow_unsup
        self.vocab = new_vocab

        self.train_sup_x, self.valid_x = _split_data(self.train_x, self.train_valid_split)
        self.train_sup_y, self.valid_y = _split_data(self.train_y, self.train_valid_split)

        # if self.shuf:
        #     import random
        #     shuffled_idx = range(new_bow.shape[0])
        #     random.seed(self.random_seed)
        #     random.shuffle(shuffled_idx)
        #     new_bow = new_bow[shuffled_idx, :]
        #     self.data_y = self.data_y[shuffled_idx]
        #     self.docids = self.docids[shuffled_idx]
        #     self.fold_numbers = self.fold_numbers[shuffled_idx]
        #     self.shuffled_idx = shuffled_idx
        #     self.raw_text = [self.raw_text[i] for i in shuffled_idx]






if __name__ == "__main__":

    categories = ['alt.atheism', 'soc.religion.christian', 'comp.windows.x', 'comp.graphics','rec.sport.baseball', 'sci.crypt']
    sixgroups = NewsGroupsData(output_dir='./20newsgroups_6cat', categories=categories)
    sixgroups.get_raw_data()
    sixgroups.get_matrices()
    sixgroups.save_and_print_data()
    # sixgroups.print_text_file()

    categories = ['alt.atheism', 'soc.religion.christian']
    bin1 = NewsGroupsData(output_dir='./20newsgroups_bin1', categories=categories)
    bin1.get_raw_data()
    bin1.get_matrices()
    bin1.save_and_print_data()
    # bin1.print_text_file()

    categories = ['comp.windows.x', 'comp.graphics']
    bin2 = NewsGroupsData(output_dir='./20newsgroups_bin2', categories=categories)
    bin2.get_raw_data()
    bin2.get_matrices()
    bin2.save_and_print_data()
    # bin2.print_text_file()

    categories = ['rec.sport.baseball', 'sci.crypt']
    bin3 = NewsGroupsData(output_dir='./20newsgroups_bin3', categories=categories)
    bin3.get_raw_data()
    bin3.get_matrices()
    bin3.save_and_print_data()
    # bin3.print_text_file()







