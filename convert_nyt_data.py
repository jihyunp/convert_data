"""
Convert NYT data into aclImdb data format
"""
from scipy import io
from scipy.sparse import csc_matrix, lil_matrix, coo_matrix, find
import numpy as np
import os
from datetime import datetime


def taxon2newcat(taxon, depth):

    if depth < 1:
        raise('depth should be larger than 0!')

    C = taxon.split('/')
    taxon_length = len(C)
    # Ignore the 0th one, which is always 'top'
    category = C[1]
    
    if depth > 1 and taxon_length > (depth-1):
        for i in range(2, depth):
            category += '_' + C[i]
       
    return category


def delete_row_lil(mat, i):
    if not isinstance(mat, lil_matrix):
        raise ValueError("works only for LIL format -- use .tolil() first")
    mat.rows = np.delete(mat.rows, i)
    mat.data = np.delete(mat.data, i)
    mat._shape = (mat._shape[0] - 1, mat._shape[1])



def map_taxon2cat(taxon, doctaxon, depth=3):

    print('Using depth = '+ str(depth)+ ' to generate categories from taxons' )
    taxon_arr = []
    oldid2newcat = []
    new_id2cat = []
    new_cat2id = {}
    newcat_id = 0
    for i, item in enumerate(taxon):
        taxon_str = str(item[0][0])
        taxon_arr.append(taxon_str)
        new_cat = taxon2newcat(taxon_str, depth)
        if new_cat == 'opinion_opinion':
            new_cat = 'opinion'
        if new_cat not in new_id2cat:
            new_id2cat.append(new_cat)
            new_cat2id[new_cat] = newcat_id
            newcat_id += 1
        oldid2newcat.append(new_cat)

    oldid2newid = []
    for i, item in enumerate(oldid2newcat):
        oldid2newid.append(new_cat2id[item])

    new_doctaxon = get_new_doctaxon(doctaxon, oldid2newid, len(new_id2cat))
    # Make it binary
    new_doctaxon_bin = (new_doctaxon != 0).astype(int)

    return new_id2cat, new_cat2id, oldid2newcat, oldid2newid, new_doctaxon_bin


def get_new_doctaxon(doctaxon, oldid2newid, num_cat):
    print('Generating new doc X taxon matrix with the new set of categories')

    num_docs = doctaxon.shape[0]
    new_doctaxon = lil_matrix((num_docs, num_cat), dtype=np.int32)

    for old_id, new_id in enumerate(oldid2newid):
        if new_id is not None:
            new_doctaxon[:,new_id] = new_doctaxon[:,new_id].toarray() + doctaxon[:, old_id].toarray()
    return new_doctaxon


def get_lab_unlab_doc_idx(oldid2newid, new_id2cat, doctaxon):

    print('Re-generating doc X taxon')
    # new_id2cat = cats_tobe_kept
    # new_cat2id = [old_cat2id[cat] for cat in cats_tobe_kept]

    num_cat = len(new_id2cat)
    new_doctaxon = get_new_doctaxon(doctaxon, oldid2newid, num_cat)
    new_doctaxon_bin = (new_doctaxon != 0).astype(int)
    new_doctaxon_bin = new_doctaxon_bin.tolil()

    label_docids = find(new_doctaxon_bin.sum(axis=1)==1)[0]
    nolabel_docids = find(new_doctaxon_bin.sum(axis=1)==0)[0]
    multilabel_docids = find(new_doctaxon_bin.sum(axis=1)>1)[0]
    nolabel_docids = np.sort(np.concatenate((nolabel_docids, multilabel_docids)))

    # for doc in nolabel_docids[::-1]:
    #     delete_row_lil(new_doctaxon_bin, doc)
    return (label_docids, nolabel_docids, new_doctaxon_bin)



def print_vocab(output_file, docword, vocab_array):
    """

    Parameters
    ----------
    output_file : output file name
    docword : bow sparse matrix
    vocab_array :

    Returns
    -------
    List[Str] : ordered vocabulary

    """
    # Sort in descending order
    vocab_output_file = os.path.join(output_folder, 'nyt.vocab')
    sorted_vocab_idx = np.argsort(np.array(docword.sum(axis=0))[0])[::-1]
    output_vocab_file = open(output_file, 'w')
    for word_idx in sorted_vocab_idx:
        word = str(vocab_array[word_idx][0][0])
        output_vocab_file.write(word + '\n')
    output_vocab_file.close()

    return sorted_vocab_idx



def get_remove_category_list(tmp_doctaxon, tmp_id2cat):
    print('Creating the category lists that should be removed')
    # Remove the categories that aren't assigned many times
    catids_tobe_removed_tmp = np.array(np.argsort(tmp_doctaxon.sum(axis=0)))[0][:13]
    cats_tobe_removed = [tmp_id2cat[i] for i in catids_tobe_removed_tmp]
    # remove 'news_front page', 'news', which are general categories
    cats_tobe_removed.append('news')
    cats_tobe_removed.append('news_front page')
    cats_tobe_removed.append('classifieds_job market')
    cats_tobe_removed.append('news_business')
    cats_tobe_removed.append('news_education')
    cats_tobe_removed.append('news_world')

    return cats_tobe_removed




def remove_combine_categories(old_id2cat, cats_tobe_removed):
    print('Remove and combining the categories..')
    newid = 0
    oldid2newid = []
    new_id2cat = []
    new_cat2id = {}

    for oldid, cat in enumerate(old_id2cat):
        if cat in cats_tobe_removed:
            oldid2newid.append(None)
        elif cat == 'news_science' or cat == 'news_technology':
            new_cat = 'news_science_technology'
            if new_cat in new_id2cat:
                newcatid = new_cat2id[new_cat]
                oldid2newid.append(newcatid)
            else:  # Seen for the first time, add it to the list
                new_id2cat.append(new_cat)
                new_cat2id[new_cat] = newid
                oldid2newid.append(newid)
                newid += 1
        elif cat == 'news_u.s.' or cat == 'news_washington' or cat == 'news_new york and region':
            new_cat = 'news_u.s.'
            if new_cat in new_id2cat:
                newcatid = new_cat2id[new_cat]
                oldid2newid.append(newcatid)
            else:  # Seen for the first time, add it to the list
                new_id2cat.append(new_cat)
                new_cat2id[new_cat] = newid
                oldid2newid.append(newid)
                newid += 1
        elif cat == 'features_arts' or cat == 'features_travel':
            new_cat = 'features_arts_travel'
            if new_cat in new_id2cat:
                newcatid = new_cat2id[new_cat]
                oldid2newid.append(newcatid)
            else:  # Seen for the first time, add it to the list
                new_id2cat.append(new_cat)
                new_cat2id[new_cat] = newid
                oldid2newid.append(newid)
                newid += 1
        else:
            new_id2cat.append(cat)
            new_cat2id[cat] = newid
            oldid2newid.append(newid)
            newid += 1
    return new_id2cat, new_cat2id, oldid2newid


def print_text(doc_ids, txt_data_folder, dst_folder):

    import subprocess
    print('Copying text files ')
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for docid in doc_ids:
        file_name = str(docid) + '.txt'
        src_file = os.path.join(txt_data_folder, file_name)
        dst_file = os.path.join(dst_folder, file_name)
        subprocess.call(['cp', src_file, dst_file])


def nyt_train_test_split_and_print_data(docword, doctaxon, label_docids, nolabel_docids, text_data_folder, output_folder, split=(1, 1)):

    import random
    from copy import copy
    import cPickle as cp

    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    train_bow_file = os.path.join(output_folder, 'train', 'labeledBow.feat')
    test_bow_file = os.path.join(output_folder, 'test', 'labeledBow.feat')
    unsup_bow_file = os.path.join(output_folder, 'train', 'unsupBow.feat')
    cp_file = os.path.join(output_folder, 'dataset.pkl')

    print('Splitting data')
    num_lab_docs = len(label_docids)
    num_unlab_docs = len(nolabel_docids)
    num_train = np.ceil(num_lab_docs * split[0] / float(split[0] + split[1]))

    shuffled_idx = copy(label_docids)
    random.seed(1234)
    random.shuffle(shuffled_idx)

    train_idx = np.sort(shuffled_idx[:num_train])
    test_idx = np.sort(shuffled_idx[num_train:])

    train_set_x_sup = docword[train_idx,:]
    train_set_y = find(doctaxon[train_idx,:])[1]

    # no valid set at this point

    test_set_x = docword[test_idx, :]
    test_set_y = find(doctaxon[test_idx,:])[1]

    train_set_x_unsup = docword[nolabel_docids,:]
    dummy = []

    datasets = ((train_set_x_sup, train_set_y), (train_set_x_unsup, dummy), (test_set_x, test_set_y))
    cp.dump(datasets, open(cp_file,'wb'))

    print('Now printing..')
    trbow = open(train_bow_file, 'w')
    for j, did in enumerate(train_idx):
        lab = train_set_y[j] + 1
        bow = find(docword[did,:])
        trbow.write('{}'.format(lab))
        for i, wid in enumerate(bow[1]):
            num = int(bow[2][i])
            trbow.write(' {}:{}'.format(wid,num))
        trbow.write('\n')
    trbow.close()

    tebow = open(test_bow_file, 'w')
    for j, did in enumerate(test_idx):
        lab = test_set_y[j] + 1
        bow = find(docword[did,:])
        tebow.write('{}'.format(lab))
        for i, wid in enumerate(bow[1]):
            num = int(bow[2][i])
            tebow.write(' {}:{}'.format(wid,num))
        tebow.write('\n')
    tebow.close()

    unsupbow = open(unsup_bow_file, 'w')
    for j, did in enumerate(nolabel_docids):
        lab =  0
        bow = find(docword[did,:])
        unsupbow.write('{}'.format(lab))
        for i, wid in enumerate(bow[1]):
            num = int(bow[2][i])
            unsupbow.write(' {}:{}'.format(wid,num))
        unsupbow.write('\n')
    unsupbow.close()

    print('Printing texts')

    train_txt_folder = os.path.join(train_folder, 'txt')
    print_text(train_idx, text_data_folder, train_txt_folder)
    test_txt_folder = os.path.join(test_folder, 'txt')
    print_text(test_idx, text_data_folder, test_txt_folder)
    unsup_txt_folder = os.path.join(unsup_folder, 'txt')
    print_text(nolabel_docids, text_data_folder, unsup_txt_folder)

    return datasets


def nyt_unsup_sup_split_and_print_data(docword, doctaxon, label_docids, nolabel_docids, text_data_folder, output_folder):

    import random
    from copy import copy
    import cPickle as cp

    sup_folder = os.path.join(output_folder, 'sup')
    unsup_folder = os.path.join(output_folder, 'unsup')

    if not os.path.exists(sup_folder):
        os.makedirs(unsup_folder)

    if not os.path.exists(sup_folder):
        os.makedirs(unsup_folder)

    unsup_bow_file = os.path.join(output_folder, 'unsupBow.feat')
    sup_bow_file = os.path.join(output_folder, 'labeledBow.feat')
    cp_file = os.path.join(output_folder, 'dataset.pkl')

    x_sup = docword[label_docids, :]
    y_sup = find(doctaxon[label_docids,:])[1]
    train_set_x_unsup = docword[nolabel_docids,:]
    dummy = []

    datasets = ((x_sup, y_sup), (train_set_x_unsup, dummy))
    cp.dump(datasets, open(cp_file,'wb'))

    print('Now printing..')
    supbow = open(sup_bow_file, 'w')
    for j, did in enumerate(label_docids):
        lab = y_sup[j] + 1
        bow = find(docword[did,:])
        supbow.write('{}'.format(lab))
        for i, wid in enumerate(bow[1]):
            num = int(bow[2][i])
            supbow.write(' {}:{}'.format(wid,num))
        supbow.write('\n')
    supbow.close()

    unsupbow = open(unsup_bow_file, 'w')
    for j, did in enumerate(nolabel_docids):
        lab =  0
        bow = find(docword[did,:])
        unsupbow.write('{}'.format(lab))
        for i, wid in enumerate(bow[1]):
            num = int(bow[2][i])
            unsupbow.write(' {}:{}'.format(wid,num))
        unsupbow.write('\n')
    unsupbow.close()

    print('Printing texts')

    sup_txt_folder = os.path.join(sup_folder, 'txt')
    print_text(label_docids, text_data_folder, sup_txt_folder)
    unsup_txt_folder = os.path.join(unsup_folder, 'txt')
    print_text(nolabel_docids, text_data_folder, unsup_txt_folder)

    return datasets


if __name__ == '__main__':

    """
    I would try manually fixing the categories
    """
    train_test_split = 0


    data_folder = '.'
    mat_data = os.path.join(data_folder, 'NewYorkTimes.mat')
    print('Loading the .mat data ..')
    data = io.loadmat(mat_data)
    # vocab, concept, taxon
    # docword, docconcept, doctaxon

    taxon = data['taxon']
    doctaxon = csc_matrix(data['doctaxon'], dtype=np.int32)
    docword = data['docword']
    vocab = data['vocab']

    # Generate categories using a lower depth of given taxon
    print(datetime.now())
    depth = 3
    tmp_id2cat, tmp_cat2id, oldid2tmpcat, oldid2tmpid, tmp_doctaxon = map_taxon2cat(taxon, doctaxon, depth)

    print(datetime.now())

    cats_tobe_removed = get_remove_category_list(tmp_doctaxon, tmp_id2cat)

    print(datetime.now())
    new_id2cat, new_cat2id, oldid2newid = remove_combine_categories(tmp_id2cat, cats_tobe_removed)
    print(datetime.now())
    label_docids, nolabel_docids, new_doctaxon = get_lab_unlab_doc_idx(oldid2newid, new_id2cat, tmp_doctaxon)
    print(datetime.now())


    print('-------')

    print(datetime.now())

    output_folder = os.path.join(data_folder, 'nyt_dataset')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    vocab_output_file = os.path.join(output_folder, 'nyt.vocab')
    text_data_folder = os.path.join(data_folder, 'txt_files')

    # Print vocabulary in the most frequent order
    sorted_vocab_idx = print_vocab(vocab_output_file, docword, vocab)
    docword_v = docword[:, sorted_vocab_idx]

    if train_test_split:

        # Split train and test as given ratio
        datasets = nyt_train_test_split_and_print_data(docword_v, new_doctaxon, label_docids, nolabel_docids, text_data_folder, output_folder, split=(1, 1))

        train_set_x_sup, train_set_y = datasets[0]
        train_set_x_unsup, _ = datasets[1]
        test_set_x, test_set_y = datasets[2]

    else:

        # Print vocabulary in the most frequent order
        sorted_vocab_idx = print_vocab(vocab_output_file, docword, vocab)
        docword_v = docword[:, sorted_vocab_idx]

        # Do not split train and test as given ratio
        datasets = nyt_unsup_sup_split_and_print_data(docword_v, new_doctaxon, label_docids, nolabel_docids, text_data_folder, output_folder)

        x_sup, y_sup = datasets[0]
        train_set_x_unsup, _ = datasets[1]


