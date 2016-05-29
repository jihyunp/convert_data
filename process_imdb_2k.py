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


from ProcessData import ProcessData
from ProcessData import _split_data

class SmallImdbData(ProcessData):

    def __init__(self, orig_data_dir, output_dir, train_valid_split=(8, 2), sup_unsup_split=(1, 2),
                 train_test_split=(8, 2), shuffle=True, random_seed=1234):
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

        ProcessData.__init__(self, orig_data_dir, output_dir, train_valid_split, sup_unsup_split, train_test_split, shuffle, random_seed)
        self.binary = True



    def _download_and_uncompress(self, url):
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
        self._download_and_uncompress(url)

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

        self.train_x, self.test_x = _split_data(new_bow,  self.train_test_split)
        self.train_y, self.test_y = _split_data(self.data_y, self.train_test_split)
        self.train_docids, self.test_docids = _split_data(self.docids, self.train_test_split)
        self.train_raw_text, self.test_raw_text= _split_data(self.raw_text, self.train_test_split)
        self.train_sup_x, self.unsup_x = _split_data(self.train_x, self.sup_unsup_split)
        self.train_sup_y, self.unsup_y = _split_data(self.train_y, self.sup_unsup_split)
        self.train_sup_docids, self.unsup_docids = _split_data(self.train_docids, self.sup_unsup_split)
        self.train_sup_raw_text, self.unsup_raw_text= _split_data(self.train_raw_text, self.sup_unsup_split)
        self.train_sup_x, self.valid_x = _split_data(self.train_sup_x, self.train_valid_split)
        self.train_sup_y, self.valid_y = _split_data(self.train_sup_y, self.train_valid_split)
        self.train_sup_docids, self.valid_docids = _split_data(self.train_sup_docids, self.train_valid_split)
        self.train_sup_raw_text, self.valid_raw_text= _split_data(self.train_sup_raw_text, self.train_valid_split)

        self.vocab = new_vocab



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


    #
    # def print_text_file(self, output_folder='./imdb_2k'):
    #     '''
    #     This is for running https://github.com/hiyijian/doc2vec
    #
    #     Parameters
    #     ----------
    #     output_folder
    #
    #     Returns
    #     -------
    #
    #     '''
    #
    #     print('Printing text file in doc2v')
    #     # Now printing text
    #     text_output_file = os.path.join(output_folder, 'text.txt')
    #     outfile = open(text_output_file, 'w')
    #     for did, text in zip(self.docids, self.raw_text):
    #         line = '_*' + str(did) + ' ' + text + '\n'
    #         outfile.write(line.encode("utf-8"))
    #     outfile.close()
    #




if __name__ == "__main__":


    obj = SmallImdbData(orig_data_dir='./imdb_2k_orig', output_dir='./imdb_2k', )
    obj.get_raw_data()
    obj.get_matrices()
    obj.save_and_print_data()
    obj.print_text_file()
    obj.print_ids_foldnums()








