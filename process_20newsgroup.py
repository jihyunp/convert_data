

from scipy import io
from scipy.sparse import csc_matrix, lil_matrix, coo_matrix, find
import numpy as np
import os
from datetime import datetime
import gzip
import re
import urllib
import tarfile
from process_imdb_2k import SmallImdb


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

class NewsGroupsData(SmallImdb):

    def __init__(self, orig_data_dir='./', output_dir='./20newsgroups', categories=[], shuffle=False, random_seed=1234):
        '''

        Parameters
        ----------
        orig_data_dir: directory where the dataset is going to be downloaded and unpacked
        output_dir: output directory that will be created
        shuffle: shuffle the indices or not
        random_seed

        '''
        SmallImdb.__init__(self, orig_data_dir=orig_data_dir, output_dir=output_dir)
        self.orig_data_dir = orig_data_dir
        self.output_dir = output_dir
        self.shuf = shuffle
        self.random_seed = random_seed

        if self.shuf:
            self.shuffled_idx = None

        self.train_raw_text = None
        self.test_raw_text = None
        self.unsup_raw_text = None

        self.train_docids = None
        self.test_docids = None
        self.unsup_docids = None

        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.unsup_x = None

        self.vocab = None

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



    def print_svmlight_format(self, x_data, y_data, bow_txt_file, unsup=False):
        print('Printing BOW ..')
        f = open(bow_txt_file, 'w')
        for i, catid in enumerate(y_data):
            if unsup:
                lab = str(int(catid))
            else:
                if self.binary:
                    if catid == 0:
                        lab = '-1'
                    else: # catid == 1
                        lab = '+1'
                else:
                    lab = str(int(catid + 1))
            f.write('{}'.format(lab))
            bow = find(x_data[i, :])
            for num, wid in zip(bow[2], bow[1]):
                num = int(num)
                f.write(' {}:{}'.format(wid+1, num))
            f.write('\n')
        f.close()


    def print_vocab(self, output_file='20ng.vocab'):
        print('Printing vocabulary ')
        # Sort in descending order
        output_vocab_file = open(output_file, 'w')
        for word in self.vocab:
            output_vocab_file.write(word + '\n')
        output_vocab_file.close()


    def save_and_print_data(self):
        # from copy import copy
        import cPickle as cp

        train_folder = os.path.join(self.output_dir, 'train')
        test_folder = os.path.join(self.output_dir, 'test')

        if not os.path.isdir(train_folder):
            os.makedirs(train_folder)

        if not os.path.isdir(test_folder):
            os.makedirs(test_folder)

        train_bow_file = os.path.join(train_folder, 'labeledBow.feat')
        unsup_bow_file = os.path.join(train_folder, 'unsupBow.feat')
        test_bow_file = os.path.join(test_folder, 'labeledBow.feat')

        cp_file = os.path.join(self.output_dir, 'dataset.pkl')

        dataset = ((self.train_x, self.train_y), (self.unsup_x, []), (self.test_x, self.test_y))

        print('Saving X matrices and label arrays to dataset.pkl ..')
        cp.dump(dataset, open(cp_file, 'wb'), protocol=cp.HIGHEST_PROTOCOL)

        self.print_svmlight_format(self.train_x, self.train_y, train_bow_file)
        self.print_svmlight_format(self.unsup_x, np.zeros(self.unsup_x.shape[0], dtype=np.int8), unsup_bow_file, unsup=True)
        self.print_svmlight_format(self.test_x, self.test_y, test_bow_file)

        # Printing vocabulary
        self.print_vocab(self.output_dir + '/20ng.vocab')



    def print_text_file(self, output_folder=None):
        '''
        This is for https://github.com/hiyijian/doc2vec
        '''

        def print_text(text_data, doc_ids, output_file):
            '''

            Parameters
            ----------
            text_data
            doc_ids
            output_file

            Returns
            -------

            '''
            outfile = open(output_file, 'w')
            for did, text in zip(doc_ids, text_data):
                line = '_*' + str(did) + ' ' + text + '\n'
                outfile.write(line.encode("utf-8"))
            outfile.close()

        if output_folder is not None:
            if output_folder != self.output_dir:
                self.output_dir = output_folder

        train_folder = os.path.join(self.output_dir, 'train')
        test_folder = os.path.join(self.output_dir, 'test')

        if not os.path.isdir(train_folder):
            os.makedirs(train_folder)

        if not os.path.isdir(test_folder):
            os.makedirs(test_folder)

        print('Printing text file in doc2v')
        # Now printing text
        train_output_file = os.path.join(train_folder, 'train_text.txt')
        print_text(self.train_raw_text, self.train_docids, train_output_file)

        unsup_output_file = os.path.join(train_folder, 'unsup_text.txt')
        print_text(self.unsup_raw_text, self.unsup_docids, unsup_output_file)

        test_output_file = os.path.join(test_folder, 'test_text.txt')
        print_text(self.test_raw_text, self.test_docids, test_output_file)



if __name__ == "__main__":

    # categories = ['alt.atheism', 'soc.religion.christian', 'comp.windows.x', 'comp.graphics','rec.sport.baseball', 'sci.crypt']
    # sixgroups = NewsGroupsData(output_dir='./20newsgroups_6cat', categories=categories)
    # sixgroups.get_raw_data()
    # sixgroups.get_matrices()
    # sixgroups.save_and_print_data()
    # sixgroups.print_text_file()

    categories = ['alt.atheism', 'soc.religion.christian']
    bin1 = NewsGroupsData(output_dir='./20newsgroups_bin1', categories=categories)
    bin1.get_raw_data()
    bin1.get_matrices()
    bin1.save_and_print_data()
    bin1.print_text_file()

    categories = ['comp.windows.x', 'comp.graphics']
    bin2 = NewsGroupsData(output_dir='./20newsgroups_bin2', categories=categories)
    bin2.get_raw_data()
    bin2.get_matrices()
    bin2.save_and_print_data()
    bin2.print_text_file()

    categories = ['rec.sport.baseball', 'sci.crypt']
    bin3 = NewsGroupsData(output_dir='./20newsgroups_bin3', categories=categories)
    bin3.get_raw_data()
    bin3.get_matrices()
    bin3.save_and_print_data()
    bin3.print_text_file()







