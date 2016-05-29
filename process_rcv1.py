


from scipy import io
from scipy.sparse import csc_matrix, lil_matrix, coo_matrix, find
import numpy as np
import os
from datetime import datetime
import gzip
import re
import urllib

from ProcessData import ProcessData
from ProcessData import _split_data

class ReutersData(ProcessData):

    def __init__(self, orig_data_dir, output_dir, train_valid_split=(8, 2), sup_unsup_split=(10, 300),
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

        ProcessData.__init__(self, orig_data_dir, output_dir, train_valid_split, sup_unsup_split, train_test_split, shuffle, random_seed)
        self.topic_raw_data = None
        self.topic_new_data = None


    #
    # def download_and_uncompress(self, url):
    #     file_name = url.split('/')[-1]
    #     file_path = os.path.join(self.orig_data_dir, file_name)
    #
    #     if not os.path.isfile(file_path):
    #         print("Downloading " + file_name + " from " + url + " ...")
    #         urllib.urlretrieve(url, file_path)
    #
    #     if file_name.split('.')[-1] == 'gz':
    #         print("Un-compressing data " + file_name)
    #         zip = gzip.open(file_path, 'rb')
    #         content = zip.read()
    #     else:
    #         f = open(file_path, 'r')
    #         content = f.read()
    #     return content


    def get_raw_data(self):
        '''

        Returns
        -------

        '''

        unsup_url = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt0.dat.gz'
        test_urls = ['http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt1.dat.gz']
        # test_urls.append('http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt2.dat.gz')
        # test_urls.append('http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt3.dat.gz')
        train_url = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_train.dat.gz'

        # 103 RCV1 Topics categories
        topics_url = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a01-list-of-topics/rcv1.topics.txt'

        # Topic hierarchy
        topic_hier_url = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a02-orig-topics-hierarchy/rcv1.topics.hier.orig'

        # specifies which Topic categories each RCV1-v2 document belongs to.
        topic_doc_url = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz'

        import urllib
        if not os.path.isdir(self.orig_data_dir):
            os.makedirs(self.orig_data_dir)

        # Get test raw data
        print('Processing test data')
        test_data = []
        test_did = []
        # Download the dataset
        for url in test_urls:
            content = self._download_and_uncompress(url)
            content_arr = re.sub('\n', ' ', content.strip()).split('.I ')[1:]
            for doc in content_arr:
                id_and_doc = doc.strip().split(' .W ')
                test_data.append(id_and_doc[1])
                test_did.append(int(id_and_doc[0]))

        # Get the train raw data
        print('Processing train data')
        train_data = []
        train_did = []
        content = self._download_and_uncompress(train_url)
        content_arr = re.sub('\n', ' ', content.strip()).split('.I ')[1:]
        for doc in content_arr:
            id_and_doc = doc.strip().split(' .W ')
            train_data.append(id_and_doc[1])
            train_did.append(int(id_and_doc[0]))

        # Get the train raw data
        print('Processing unlabeled training data')
        unsup_data = []
        unsup_did = []
        content = self._download_and_uncompress(unsup_url)
        content_arr = re.sub('\n', ' ', content.strip()).split('.I ')[1:]
        for doc in content_arr:
            id_and_doc = doc.strip().split(' .W ')
            unsup_data.append(id_and_doc[1])
            unsup_did.append(int(id_and_doc[0]))


        # Get topic data
        print('Processing topic information')
        td_content = self._download_and_uncompress(topic_doc_url)
        td_content_arr = td_content.strip().split('\n')
        tid2topic = []
        topic2tid = {}
        did2tid = {}
        tid = 0
        for line in td_content_arr:
            topic = line.split(' ')[0]
            docid = int(line.split(' ')[1])
            if topic not in tid2topic:
                tid2topic.append(topic)
                topic2tid[topic] = tid
                did2tid[docid] = tid
                tid += 1
            else:
                tid2 = topic2tid[topic]
                did2tid[docid] = tid2

        # Get topic hierarchies
        th_content = self._download_and_uncompress(topic_hier_url)
        th_content_arr = th_content.strip().split('\n')
        topic_child2parent = {}
        topic_desc = {}
        for line in th_content_arr:
            sp_line = re.split('\s+', line)
            parent = sp_line[1]
            child = sp_line[3]
            child_dsc = sp_line[5]
            topic_child2parent[child] = parent
            topic_desc[child] = child_dsc

        self.topic_raw_data = (tid2topic, topic2tid, did2tid, topic_child2parent, topic_desc)

        self.train_raw_text = train_data
        self.train_docids = train_did
        self.test_raw_text = test_data
        self.test_docids = test_did
        self.unsup_raw_text = unsup_data
        self.unsup_docids = unsup_did




    def map_parent_categories(self):
        '''

        Parameters
        ----------
        topic_raw
            (old_id2cat, old_cat2id, old_did2cid, topic_child2parent, old_topic_desc)
        Returns
        -------

        '''
        print('Mapping child to parent categories (topics).. Generating new set of \'topic_raw\'')
        old_id2cat, old_cat2id, old_did2cid, topic_child2parent, old_topic_desc = self.topic_raw_data

        newid = 0
        oldtid2newtid = []
        new_id2cat = []
        new_cat2id = {}
        new_topic_desc = []
        new_did2tid = {}

        for child in topic_child2parent:
            parent = topic_child2parent[child]
            if parent == 'Root':
                new_id2cat.append(child)
                new_cat2id[child] = newid
                new_topic_desc.append(old_topic_desc[child])
                newid += 1

        for oldid, child in enumerate(old_id2cat):
            parent = child
            while parent not in new_id2cat:
                tmp = topic_child2parent[child]
                if tmp == 'None' or tmp == 'Root':
                    break
                parent = tmp
                child = parent
            newid2 = new_cat2id[parent]
            oldtid2newtid.append(newid2)

        # get did2tid
        for did in old_did2cid:
            old_cid = old_did2cid[did]
            newtid = oldtid2newtid[old_cid]
            new_did2tid[did] = newtid

        self.topic_new_data = (new_id2cat, new_cat2id, new_did2tid, topic_child2parent, new_topic_desc)

    def get_matrices(self): #train_raw, test_raw, topic_raw, sup_unsup_ratio=(1, 30)):
        '''

        Parameters
        ----------
        train_raw : Tup( List[str], List[int] )

        test_raw : Tup( List[str], List[int] )

        topic_raw  : (list[str], dict[str:int], dict[int:int], dict[str:str], dict[str:str])
                (tid2topic, topic2tid, did2tid, topic_child2parent, topic_desc)


        Returns
        -------
        bow and corresponding doc x category matrix

        '''
        print('Generating BOW matrices for train/test set')
        from sklearn.feature_extraction.text import CountVectorizer

        # Get BOWs for train/test
        train_text = self.train_raw_text
        train_did = self.train_docids
        unsup_text = self.unsup_raw_text
        unsup_did = self.unsup_docids
        test_text = self.test_raw_text
        test_did = self.test_docids

        count_vec = CountVectorizer()
        count_vec = count_vec.fit(train_text + unsup_text)
        train_bow = count_vec.transform(train_text)
        unsup_bow = count_vec.transform(unsup_text)
        test_bow = count_vec.transform(test_text)
        vocab = count_vec.vocabulary_
        vocab_rev = {y: x for x, y in vocab.iteritems()}

        # Change vocab order
        sorted_vocab_idx = np.argsort(np.array(train_bow.sum(axis=0))[0])[::-1]
        self.train_x = train_bow[:, sorted_vocab_idx]
        self.unsup_x = unsup_bow[:, sorted_vocab_idx]
        self.test_x = test_bow[:, sorted_vocab_idx]
        new_vocab = [vocab_rev[i] for i in sorted_vocab_idx]
        self.vocab = new_vocab

        print('Generating labels for train/test set')
        # Get topic labels for train/test set
        did2tid = self.topic_new_data[2]
        self.train_y = np.array([did2tid[did] for did in train_did], dtype=np.int16)
        self.test_y = np.array([did2tid[did] for did in test_did], dtype=np.int16)

        # SPlit the data
        self.train_sup_x, self.valid_x = _split_data(self.train_x, self.train_valid_split)
        self.train_sup_y, self.valid_y = _split_data(self.train_y, self.train_valid_split)
        self.train_sup_docids, self.valid_docids = _split_data(self.train_docids, self.train_valid_split)

        # return ((train_bow, train_y, train_did), (test_bow, test_y, test_did), new_vocab)


    def print_topics(self, output_file):

        id2cat = self.topic_new_data[0]
        id2desc = self.topic_new_data[4]

        print('Printing topic (category) information')
        outfile = open(output_file, 'w')
        id = 0
        for cat, desc in zip(id2cat, id2desc):
            outfile.write('{}\t{}\t{}\n'.format(id, cat, desc))
        outfile.close()






if __name__ == "__main__":
   obj = ReutersData('./rcv1-v2_orig', './rcv1-v2')
   obj.get_raw_data()
   obj.map_parent_categories()
   obj.get_matrices()
   obj.save_and_print_data()
   obj.print_topics(obj.output_dir + '/rcv1.topics')
   obj.print_text_file()

