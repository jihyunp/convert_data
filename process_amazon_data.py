import gzip
import os
import urllib
import re
import json

import numpy as np
from scipy.sparse import find

from ProcessData import ProcessData
from ProcessData import _split_data

import random


class AmazonData(ProcessData):

    def __init__(self, orig_data_dir, output_dir, categories, class_limit=10000, train_valid_split=(8,2), sup_unsup_split=(1,10),
                       train_test_split=(8,2), shuffle=True, random_seed=1234):
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

        ProcessData.__init__(self, orig_data_dir, output_dir, train_valid_split, sup_unsup_split, train_test_split, shuffle,
                             random_seed)

        self.class_limit = class_limit
        self.categories = categories
        self.all_categories = ['clothing', 'home', 'toys', 'cellphones', 'health', 'sports' ]

        self.binary = False
        if len(categories) == 2:
            self.binary = True

    def print_all_categories(self):
        print(self.all_categories)

    def print_selected_categories(self):
        print(self.categories)

    def _download_and_uncompress(self, url):
        file_name = url.split('/')[-1]
        file_path = os.path.join(self.orig_data_dir, file_name)

        if not os.path.isfile(file_path):
            print("Downloading " + file_name + " from " + url + " ...")
            urllib.urlretrieve(url, file_path)

        txt_arr = []
        reviewer_ids = []
        if file_name.split('.')[-1] == 'gz':
            print("Un-compressing data " + file_name)
            zip = gzip.open(file_path, 'rb')
            for line in zip:
                tmp_dict = json.loads(line)
                txt_arr.append(tmp_dict['reviewText'])
                reviewer_ids.append(tmp_dict['reviewerID'])
        else:
            print('Doing nothing.. returning empty lists')

        return (txt_arr, reviewer_ids)


    def get_raw_data(self):

        urls = []
        if 'clothing' in self.categories:
            urls.append('http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json.gz')
        if 'home' in self.categories:
            urls.append('http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Home_and_Kitchen_5.json.gz')
        if 'toys' in self.categories:
            urls.append('http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz')
        if 'cellphones' in self.categories:
            urls.append('http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz')
        if 'health' in self.categories:
            urls.append('http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Health_and_Personal_Care_5.json.gz')
        if 'sports' in self.categories:
            urls.append('http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz')

        if not os.path.isdir(self.orig_data_dir):
            os.makedirs(self.orig_data_dir)

        txt_list = []
        labels_arr = np.array([], dtype=np.int8)
        uid_list  = []

        for i, url in enumerate(urls):
            txt_list_tmp , uid_list_tmp = self._download_and_uncompress(url)
            # Take 10,000(class_limit) samples from each categories
            num_reviews = len(uid_list_tmp)
            random.seed(self.random_seed)  # So that we can get the same split
            sample_idx = random.sample(range(num_reviews), self.class_limit)
            txt_list_tmp = [txt_list_tmp[k] for k in sample_idx]
            uid_list_tmp = [uid_list_tmp[k] for k in sample_idx]

            for j, txt in enumerate(txt_list_tmp):
                txt2 = txt.lower()
                txt2 = re.sub('[^a-z0-9\-\' ]', '', txt2)
                txt2 = re.sub('\-\-', ' ', txt2)
                txt2 = re.sub(' - ', ' ', txt2)
                txt2 = re.sub(r'\s+', ' ', txt2)  # removes white spaces
                txt_list_tmp[j] = txt2

            labels_tmp = np.ones(len(uid_list_tmp), dtype=np.int8) * i
            txt_list = txt_list + txt_list_tmp
            uid_list = uid_list + uid_list_tmp
            labels_arr = np.concatenate((labels_arr, labels_tmp))

        self.raw_text = txt_list
        self.data_y = labels_arr
        self.docids = np.array(range(len(labels_arr)), dtype=np.int32)

        if len(np.unique(labels_arr)) == 2:
            self.binary = True

        if self.shuf:
            shuffled_idx = range(len(labels_arr))
            random.seed(self.random_seed)  # Use the same seed
            random.shuffle(shuffled_idx)
            self.data_y = self.data_y[shuffled_idx]
            self.docids = self.docids[shuffled_idx]
            self.shuffled_idx = shuffled_idx
            self.raw_text = [self.raw_text[i] for i in shuffled_idx]

        self.train_raw_text, self.test_raw_text= _split_data(self.raw_text, self.train_test_split)
        self.train_y, self.test_y = _split_data(self.data_y, self.train_test_split)
        self.train_docids, self.test_docids = _split_data(self.docids, self.train_test_split)



    def get_matrices(self):

        print('Generating BOW matrix')
        from sklearn.feature_extraction.text import CountVectorizer
        from nltk.corpus import stopwords

        count_vec = CountVectorizer(stop_words=stopwords.words('english'))
        count_vec = count_vec.fit(self.train_raw_text)
        self.train_x = count_vec.transform(self.train_raw_text)
        self.test_x = count_vec.transform(self.test_raw_text)
        vocab = count_vec.vocabulary_
        vocab_inv = {y:x for x,y in vocab.iteritems()}

        # Change vocab order
        sorted_vocab_idx = np.argsort(np.array(self.train_x.sum(axis=0))[0])[::-1]
        self.train_x = self.train_x[:,sorted_vocab_idx]
        self.test_x = self.test_x[:,sorted_vocab_idx]
        new_vocab = [vocab_inv[i] for i in sorted_vocab_idx]

        self.train_sup_x, self.unsup_x = _split_data(self.train_x, self.sup_unsup_split)
        self.train_sup_y, self.unsup_y = _split_data(self.train_y, self.sup_unsup_split)
        self.train_sup_docids, self.unsup_docids = _split_data(self.train_docids, self.sup_unsup_split)
        self.train_sup_raw_text, self.unsup_raw_text= _split_data(self.train_raw_text, self.sup_unsup_split)
        self.train_sup_x, self.valid_x = _split_data(self.train_sup_x, self.train_valid_split)
        self.train_sup_y, self.valid_y = _split_data(self.train_sup_y, self.train_valid_split)
        self.train_sup_docids, self.valid_docids = _split_data(self.train_sup_docids, self.train_valid_split)
        self.train_sup_raw_text, self.valid_raw_text= _split_data(self.train_sup_raw_text, self.train_valid_split)

        self.vocab = new_vocab




if __name__ == "__main__":

    cat = ['clothing', 'home', 'toys', 'cellphones', 'health', 'sports' ]
    mult = AmazonData(orig_data_dir='./amazon_orig', output_dir='./amazon_6cat', categories=cat)
    mult.get_raw_data()
    mult.get_matrices()
    mult.save_and_print_data()
    mult.print_text_file()

    cat = ['clothing', 'home']
    bin1 = AmazonData(orig_data_dir='./amazon_orig', output_dir='./amazon_bin1', categories=cat)
    bin1.get_raw_data()
    bin1.get_matrices()
    bin1.save_and_print_data()
    bin1.print_text_file()

    cat = ['toys', 'cellphones']
    bin2 = AmazonData(orig_data_dir='./amazon_orig', output_dir='./amazon_bin2', categories=cat)
    bin2.get_raw_data()
    bin2.get_matrices()
    bin2.save_and_print_data()
    bin2.print_text_file()

    cat = ['health', 'sports' ]
    bin3 = AmazonData(orig_data_dir='./amazon_orig', output_dir='./amazon_bin3', categories=cat)
    bin3.get_raw_data()
    bin3.get_matrices()
    bin3.save_and_print_data()
    bin3.print_text_file()


