import gzip
import os
import urllib

import numpy as np
from scipy.sparse import find


def _split_data(data, split):
    if type(data) == list:
        N = len(data)
    else:
        N = data.shape[0]
    if sum(split) != N:
        const = int(np.ceil(N / float(sum(split))))
    else:
        const = 1
    starts = np.cumsum(np.r_[0, split[:-1]] * const)
    ends = np.cumsum(split) * const
    if ends[-1] > N:
        ends[-1] = N
    splits = [data[s:e] for s, e in zip(starts, ends)]
    return splits


class ProcessData():

    def __init__(self, orig_data_dir, output_dir, train_valid_split=(8,2), sup_unsup_split=(10,300),
                       train_test_split=(8,2), shuffle=False, random_seed=1234):
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
        self.orig_data_dir = orig_data_dir
        self.output_dir = output_dir

        self.shuf = shuffle
        self.random_seed = random_seed

        if self.shuf:
            self.shuffled_idx = None

        self.train_raw_text = None

        self.train_sup_raw_text = None
        self.test_raw_text = None
        self.valid_raw_text = None
        self.unsup_raw_text = None

        self.train_docids = None

        self.train_sup_docids = None
        self.valid_docids = None
        self.test_docids = None
        self.unsup_docids = None

        self.train_x = None
        self.train_y = None

        self.train_sup_x = None
        self.train_sup_y = None
        self.valid_x = None
        self.valid_y = None
        self.test_x = None
        self.test_y = None
        self.unsup_x = None

        self.train_valid_split = train_valid_split
        self.sup_unsup_split = sup_unsup_split
        self.train_test_split = train_test_split
        self.vocab = None

        self.binary = False
        # self.categories = categories
        # self.all_categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
        #                        'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
        #                        'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
        #                        'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
        #                        'talk.politics.misc', 'talk.religion.misc']
        #
        # self.binary=False
        # if len(categories) == 2:
        #     self.binary=True



    def get_raw_data(self):
        '''
            Function to get the raw_texts, y labels, and docids
            for train/unsup/test set. (not validation set yet)
        '''
        pass

    def _download_and_uncompress(self, url):
        '''
        Used inside the 'get_raw_data' function if needed.

        Returns
        -------
        String, or a list of string

        '''
        file_name = url.split('/')[-1]
        file_path = os.path.join(self.orig_data_dir, file_name)

        if not os.path.isfile(file_path):
            print("Downloading " + file_name + " from " + url + " ...")
            urllib.urlretrieve(url, file_path)

        if file_name.split('.')[-1] == 'gz':
            print("Un-compressing data " + file_name)
            zip = gzip.open(file_path, 'rb')
            content = zip.read()
        else:
            f = open(file_path, 'r')
            content = f.read()
        return content

    def get_matrices(self):
        '''
        Get BOW matrices and y arrays for train_sup/valid/train_unsup/test data.
        Shuffle the matrices if needed.
        Split the training data into train_sup and valid.
        Sort the vocabularies in the order of frequency.

        Returns
        -------
        Don't have to return, just make sure you assign the variables below.

        self.train_sup_x
        self.train_sup_y
        self.valid_x
        self.valid_y
        self.unsup_x
        self.test_x
        self.test_y
        self.vocab

        '''
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

        self.train_sup_x, self.valid_x = _split_data(self.train_x, split=self.split)
        self.train_sup_y, self.valid_y = _split_data(self.train_y, split=self.split)
        self.train_sup_docids, self.valid_docids = _split_data(self.train_docids, split=self.split)

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
                f.write(' {}:{}'.format(wid+1, num)) # Word id should start from 1
            f.write('\n')
        f.close()


    def print_vocab(self, output_file='vocab'):
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
        valid_folder = os.path.join(self.output_dir, 'valid')

        if not os.path.isdir(train_folder):
            os.makedirs(train_folder)
        if not os.path.isdir(test_folder):
            os.makedirs(test_folder)
        if not os.path.isdir(valid_folder):
            os.makedirs(valid_folder)

        train_sup_bow_file = os.path.join(train_folder, 'labeledBow.feat')
        unsup_bow_file = os.path.join(train_folder, 'unsupBow.feat')
        valid_bow_file = os.path.join(valid_folder, 'labeledBow.feat')
        test_bow_file = os.path.join(test_folder, 'labeledBow.feat')

        cp_file = os.path.join(self.output_dir, 'dataset.pkl')
        dataset = ((self.train_sup_x, self.train_sup_y), (self.valid_x, self.valid_y), (self.unsup_x, []), (self.test_x, self.test_y))
        print('Saving X matrices and label arrays to dataset.pkl ..')
        cp.dump(dataset, open(cp_file, 'wb'), protocol=cp.HIGHEST_PROTOCOL)

        self.print_svmlight_format(self.train_sup_x, self.train_sup_y, train_sup_bow_file)
        self.print_svmlight_format(self.unsup_x, np.zeros(self.unsup_x.shape[0], dtype=np.int8), unsup_bow_file, unsup=True)
        self.print_svmlight_format(self.valid_x, self.valid_y, valid_bow_file)
        self.print_svmlight_format(self.test_x, self.test_y, test_bow_file)

        # Printing vocabulary
        self.print_vocab(self.output_dir + '/vocab')

    def _print_text(self, text_data, doc_ids, output_file):
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

    def print_text_file(self, output_folder=None):
        '''
        This is for https://github.com/hiyijian/doc2vec
        '''
        if output_folder is not None:
            if output_folder != self.output_dir:
                self.output_dir = output_folder

        train_folder = os.path.join(self.output_dir, 'train')
        valid_folder = os.path.join(self.output_dir, 'valid')
        test_folder = os.path.join(self.output_dir, 'test')

        if not os.path.isdir(train_folder):
            os.makedirs(train_folder)
        if not os.path.isdir(valid_folder):
            os.makedirs(valid_folder)
        if not os.path.isdir(test_folder):
            os.makedirs(test_folder)

        print('Printing text file in doc2vec form')
        # Now printing text
        train_output_file = os.path.join(train_folder, 'train_sup_text.txt')
        self._print_text(self.train_sup_raw_text, self.train_sup_docids, train_output_file)

        unsup_output_file = os.path.join(train_folder, 'train_unsup_text.txt')
        self._print_text(self.unsup_raw_text, self.unsup_docids, unsup_output_file)

        valid_output_file = os.path.join(valid_folder, 'valid_sup_text.txt')
        self._print_text(self.valid_raw_text, self.valid_docids, valid_output_file)

        test_output_file = os.path.join(test_folder, 'test_text.txt')
        self._print_text(self.test_raw_text, self.test_docids, test_output_file)





