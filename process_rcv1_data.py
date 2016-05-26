from scipy import io
from scipy.sparse import csc_matrix, lil_matrix, coo_matrix, find
import numpy as np
import os
from datetime import datetime
import gzip
import re

def get_raw_data(data_dir):

    """
    Download RCV1-v2/LYRL2004 dataset
    Readme Page: http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm

    Parameters
    ----------
    data_dir : folder path where the data should be downloaded/unpacked
           ex) '/home/datalab/data/rcv1-v2

    Returns
    -------

    """

    def download_and_uncompress(url):
        file_name = url.split('/')[-1]
        file_path = os.path.join(data_dir, file_name)

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

    test_urls = []
    test_urls.append('http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt0.dat.gz')
    test_urls.append('http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt1.dat.gz')
    test_urls.append('http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt2.dat.gz')
    test_urls.append('http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt3.dat.gz')
    train_url = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_train.dat.gz'

    # 103 RCV1 Topics categories
    topics_url = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a01-list-of-topics/rcv1.topics.txt'

    # Topic hierarchy
    topic_hier_url = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a02-orig-topics-hierarchy/rcv1.topics.hier.orig'

    # specifies which Topic categories each RCV1-v2 document belongs to.
    topic_doc_url = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz'

    import urllib
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    # Get test raw data
    print('Processing test data')
    test_data = []
    test_did = []
    # Download the dataset
    for url in test_urls:
        content = download_and_uncompress(url)
        content_arr = re.sub('\n', ' ', content.strip()).split('.I ')[1:]
        for doc in content_arr:
            id_and_doc = doc.strip().split(' .W ')
            test_data.append(id_and_doc[1])
            test_did.append(int(id_and_doc[0]))

    # Get the train raw data
    print('Processing train data')
    train_data = []
    train_did = []
    content = download_and_uncompress(train_url)
    content_arr = re.sub('\n', ' ', content.strip()).split('.I ')[1:]
    for doc in content_arr:
        id_and_doc = doc.strip().split(' .W ')
        train_data.append(id_and_doc[1])
        train_did.append(int(id_and_doc[0]))

    # Get topic data
    print('Processing topic information')
    td_content = download_and_uncompress(topic_doc_url)
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
    th_content = download_and_uncompress(topic_hier_url)
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

    topic_data = (tid2topic, topic2tid, did2tid, topic_child2parent, topic_desc)

    return ((train_data, train_did), (test_data, test_did), topic_data)


def map_parent_categories(topic_raw):
    '''

    Parameters
    ----------
    topic_raw
        (old_id2cat, old_cat2id, old_did2cid, topic_child2parent, old_topic_desc)
    Returns
    -------

    '''
    print('Mapping child to parent categories (topics).. Generating new set of \'topic_raw\'')
    old_id2cat, old_cat2id, old_did2cid, topic_child2parent, old_topic_desc = topic_raw

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

    return (new_id2cat, new_cat2id, new_did2tid, topic_child2parent , new_topic_desc)



def get_matrices(train_raw, test_raw, topic_raw):
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
    train_text = train_raw[0]
    train_did = train_raw[1]

    test_text = test_raw[0]
    test_did = test_raw[1]

    count_vec = CountVectorizer()
    train_bow = count_vec.fit_transform(train_text)
    test_bow = count_vec.transform(test_text)
    vocab = count_vec.vocabulary_
    vocab_rev = {y:x for x,y in vocab.iteritems()}

    # Change vocab order
    sorted_vocab_idx = np.argsort(np.array(train_bow.sum(axis=0))[0])[::-1]
    train_bow = train_bow[:,sorted_vocab_idx]
    test_bow = test_bow[:, sorted_vocab_idx]
    new_vocab = [vocab_rev[i] for i in sorted_vocab_idx]

    print('Generating labels for train/test set')
    # Get topic labels for train/test set
    did2tid = topic_raw[2]
    train_y = [did2tid[did] for did in train_did]
    test_y = [did2tid[did] for did in test_did]

    return ((train_bow, train_y, train_did), (test_bow, test_y, test_did), new_vocab)


def print_svmlight_format(x_sparse_mat, y_label_arr, bow_file):

    print('Printing BOW ..')
    bow_file = open(bow_file, 'w')
    for i, catid in enumerate(y_label_arr):
        lab = catid + 1
        bow = find(x_sparse_mat[i, :])
        bow_file.write('{}'.format(lab))
        for num, wid in zip(bow[2], bow[1]):
            num = int(num)
            bow_file.write(' {}:{}'.format(wid, num))
        bow_file.write('\n')
    bow_file.close()

def print_vocab(vocab_array, output_file='rcv1.vocab'):
    """

    Parameters
    ----------
    vocab_array : Array sorted from the most popular word to least popular word
    output_file : output file name

    Returns
    -------

    """
    print('Printing vocabulary ')
    # Sort in descending order
    output_vocab_file = open(output_file, 'w')
    for word in vocab_array:
        output_vocab_file.write(word + '\n')
    output_vocab_file.close()


def print_topics(categories, output_file='rcv1.topics'):

    id2cat = categories[0]
    id2desc = categories[4]

    print('Printing topic (category) information')
    outfile = open(output_file, 'w')
    id = 0
    for cat, desc in zip(id2cat, id2desc):
        outfile.write('{}\t{}\t{}\n'.format(id, cat, desc))
    outfile.close()


def save_and_print_data(train_data, test_data, vocab, output_folder='./rcv1-v2_imdb_format'):

    # from copy import copy
    import cPickle as cp

    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')

    if not os.path.isdir(train_folder):
        os.makedirs(train_folder)

    if not os.path.isdir(test_folder):
        os.makedirs(test_folder)

    train_bow_file = os.path.join(train_folder, 'labeledBow.feat')
    test_bow_file = os.path.join(test_folder, 'labeledBow.feat')
    cp_file = os.path.join(output_folder, 'dataset.pkl')

    train_x, train_y, _ = train_data
    test_x, test_y, _ = test_data

    datasets = (train_data[:2], test_data[:2])
    print('Saving train/test matrices and labels to dataset.pkl ..')
    cp.dump(datasets, open(cp_file,'wb'), protocol=cp.HIGHEST_PROTOCOL)

    print_svmlight_format(train_x, train_y, train_bow_file)
    print_svmlight_format(test_x, test_y, test_bow_file)

    # Printing vocabulary
    print_vocab(vocab, output_folder + '/rcv1.vocab')


def print_text_file(train_raw, test_raw, output_folder='./rcv1-v2_imdb_format'):
    '''
    This is for https://github.com/hiyijian/doc2vec

    Parameters
    ----------
    train_raw
    test_raw
    output_folder

    Returns
    -------

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

    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')

    if not os.path.isdir(train_folder):
        os.makedirs(train_folder)

    if not os.path.isdir(test_folder):
        os.makedirs(test_folder)

    print('Printing text file in doc2v')
    # Now printing text
    train_text = train_raw[0]
    train_docids = train_raw[1]
    train_output_file = os.path.join(train_folder, 'train_text.txt')
    print_text(train_text, train_docids, train_output_file)

    test_text = test_raw[0]
    test_docids = test_raw[1]
    test_output_file = os.path.join(test_folder, 'test_text.txt')
    print_text(test_text, test_docids, test_output_file)



if __name__ == "__main__":

    data_dir = './rcv1-v2'
    output_dir = './rcv1-v2_formatted'

    raw_dataset = get_raw_data(data_dir=data_dir)
    train_raw = raw_dataset[0]
    test_raw = raw_dataset[1]
    categories_raw = raw_dataset[2]  # Has (tid2topic, topic2tid, did2tid, topic_child2parent, topic_desc)

    # 4 topics (parents)
    new_id2cat, new_cat2id, new_did2tid, topic_child2parent , new_topic_desc = map_parent_categories(topic_raw)
    new_categories = map_parent_categories(categories_raw)

    train_data, test_data, vocab = get_matrices(train_raw, test_raw, new_categories)

    save_and_print_data(train_data, test_data, vocab, output_folder=output_dir)
    print_text_file(train_raw, test_raw, output_dir)
    print_topics(new_categories, output_file=output_dir + '/rcv1.topics')










