import copy
import errno
import os
import string
from math import log
from glob import glob
from porter2stemmer import Porter2Stemmer
import numpy as np
from scipy import stats


class Topics:
    """
    this would be topic definitions
    """

    def __init__(self, _title, _desc='', _narr=''):
        self.title = _title
        self.desc = _desc
        self.narr = _narr


class Corpus:
    """
    this would have docs of 1 complete Training set i.e. dataset101-150/Training105
    """

    def __init__(self, training_set_name):
        """
        constructor for the corpus class which holds the details of the training data
        """
        self.training_data = {}
        self.total_words = {}
        self.name = training_set_name

    def add_item(self, training_data, doc_id, total_words):
        """
        this adds a dictionary of training words and the total word counts of a document to the private members
        training_data and total_words
        """
        self.training_data[doc_id] = training_data
        self.total_words[doc_id] = total_words

    def get_avg_doc_len(self):
        """
        this returns the average document lenght of all the documents in the specific folder
        """
        return round(sum(self.total_words.values())/len(self.total_words.keys()))

    def get_total_docs_containing_term(self, term):
        """
        this returns the total number of documents which contain the term
        """
        return len([1 for val in self.training_data.values() if term in val.keys()])

    def get_count_in_relevant_docs(self, term, relevant_docs):
        """
        this returns the count of the term in the relevant_docs
        """
        return len([key for key, val in self.training_data.items() if term in val.keys() if key in relevant_docs])

    def get_total_corpus_words(self):
        """
        this returns the total number of words in the folder i.e. Training101
        """
        return sum(self.total_words.values())

    def get_total_docs(self):
        """
        this returns the total number of documents in the training folder i.e. dataset101-150/Training105
        """
        return len(self.total_words.keys())

    def get_new_vector(self):
        """
        this returns an empty list which has the exact length as the number of terms in the training folder
        i.e. dataset101-150/Training105
        """
        df = self.calc_df()
        for i in df.values():
            i = 0
        return df.fromkeys(df, 0)

    def calc_df(self):
        """
        this returns the dictionary of all terms in the document with key:value pair as term:(total count in training
        folder) i.e. dataset101-150/Training105
        """
        aux_dict = {}
        for i in self.training_data.keys():
            for j in self.training_data[i].keys():
                if j not in aux_dict.keys():
                    aux_dict[j] = 0

        temp_dict = copy.deepcopy(aux_dict)
        for i in self.training_data.keys():
            # converting all values of temp_dict to 0
            temp_dict = dict.fromkeys(temp_dict, 0)
            for j in self.training_data[i].keys():
                if temp_dict[j] == 0:
                    temp_dict[j] = 1
                    aux_dict[j] += 1
        return aux_dict


def get_set(ls, idx):
    """
    this returns the Corpus object which of which index is passed as idx
    """
    for _ls in ls:
        if str(idx) in _ls.name:
            return _ls
    return _ls


def get_stop_words() -> list:
    """
    this function reads the common stop words in the file common-english-words.txt which must be in the root directory
    :return: list of all the stop words in the folder
    """
    path_stop_list = "common-english-words.txt"
    reader = open(path_stop_list, "r")
    stop_list = reader.read()
    reader.close()
    return stop_list.split(',')


def parse_doc(file: str):
    """
    :param file: the xml file which needs to be parsed
    :return: new_dic, (docid, word_count)
        new_dic is the dictionary which has the frequency of all the terms
        docid is the id of the document for which the tf are found
        word_count is the the count of all the words in the documents
    """
    my_file = open(file)

    curr_doc = {}
    start_end = False

    file_ = my_file.readlines()
    word_count = 0
    for line in file_:
        line = line.strip()
        if start_end is False:
            if line.startswith("<newsitem "):
                for part in line.split():
                    if part.startswith("itemid="):
                        docid = part.split("=")[1].split("\"")[1]
                        break
            if line.startswith("<text>"):
                start_end = True
        elif line.startswith("</text>"):
            break
        else:
            line = line.replace("<p>", "").replace("</p>", "")
            line = line.translate(str.maketrans('', '', string.digits)).translate(
                str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
            line = line.replace("\\s+", " ")

            line = line.split()

            line = list(filter(None, line))

            # converting to str and then lowercase
            line = [te.lower() for te in line]

            # applying stemming to the document
            ps = Porter2Stemmer()
            line = map(ps.stem, line)

            for term in line:
                word_count += 1
                # wk3
                if len(term) > 2 and term not in stop_word_list:
                    # wk3
                    try:
                        curr_doc[term] += 1
                    except KeyError:
                        curr_doc[term] = 1
    my_file.close()

    new_dic = {}
    for i in curr_doc.keys():
        new_dic[i] = [i, curr_doc[i]]

    return new_dic, (docid, word_count)


def loads_dataset(name: list) -> Corpus:
    """
    :param name: the folder which has the xml files for the specific training dataset
    :return: the Corpus object which contains the information for the xml documents in certain folder i.e. Training101
    """

    training = Corpus(name.split('/')[-1])
    folder = "{}/*".format(name)
    xml_files = glob(folder)
    for file in xml_files:
        if '.txt' not in file:
            try:
                with open(file, "r", encoding='ISO-8859-1') as f:
                    html = f.read()
                    try:
                        html = html[html.index('<body'):]
                    except ValueError:
                        try:
                            html = html[html.index('<BODY'):]
                        except ValueError:
                            pass
            except IOError as exc:
                if exc.errno != errno.EISDIR:
                    raise

        temp_dict, (docID, word_count) = parse_doc(file)

        training.add_item(training_data=temp_dict, doc_id=docID, total_words=word_count)

    return training


def filtering_stemming(t: list) -> list:
    """
    this returns the list of all the preprocessed words, i.e. removing null words, converting all words to lower case,
    applying stop words from the get_stop_words function and finally stemming them using the Porter2Stemmer
    """
    # removing nulls
    t = list(filter(None, t))

    # converting to str and then lowercase
    t = [te.lower() for te in t]

    # applying stop wording to the document
    for te in list(t):
        if te in stop_word_list:
            t.remove(te)

    # applying stemming to the document
    ps = Porter2Stemmer()
    stemmed_words = map(ps.stem, t)
    return list(stemmed_words)


def bm25_baseline_model(query_list: dict, data: dict) -> dict:
    """
    this function uses the bm25 model to find the score of the query in different documents, assigning a score to the
    different documents.
    """

    # setting the constants
    k1 = 1.2
    k2 = 100  # can be changed btw 0 to 1000
    b = 0.75
    sample_list = [0] * data.get_total_docs()
    scores = {}
    mu = data.get_avg_doc_len()

    df = data.calc_df()

    flag1 = 0
    scores_temp0 = {}
    for q in query_list:
        scores_temp = {}
        for i, p in zip(range(data.get_total_docs()), data.training_data.keys()):
            temp0 = data.total_words[p] / mu
            temp0 = b * temp0
            temp0 = (1 - b) + temp0
            K = k1 * temp0

            # numerator
            temp_var3 = (1 + k2) * query_list.count(q)
            # num/denom
            temp_var3 = temp_var3 / (k2 + query_list.count(q))

            doc_freq = 0
            for i in df:
                if i[0] == q:
                    doc_freq = i[1]

            query_doc_len = 0
            for i in data.training_data[p].values():
                if q == i[0]:
                    query_doc_len = i[1]
            # numerator
            temp_var2 = (1 + k1) * query_doc_len
            # num/denom
            temp_var2 = temp_var2 / (K + query_doc_len)

            temp_var1 = len(files) + 0.5  # numerator
            temp_var1 = temp_var1 / (doc_freq + 0.5)  # num/denom
            temp_var1 = log(temp_var1, 10)  # taking log

            final_var = temp_var1 * temp_var2 * temp_var3
            scores_temp[p] = final_var

        if flag1 == 0:
            for a, b in zip(scores_temp.keys(), sample_list):
                scores_temp0[a] = scores_temp[a] + b
            flag1 = 1
        else:
            for a, b in zip(scores_temp0.keys(), scores_temp.values()):
                scores_temp0[a] = scores_temp0[a] + b

    return scores_temp0


def write_dat_files(score: dict, idx: str, folder_name: str) -> bool:
    """
    this writes bm25 model scores to files in the folder BM25 which is in the root folder
    """
    # create the BM25 folder if does not exist
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    # looping over the 50 queries and writing score files one by one
    with open('{1}/B_Result{0}.dat'.format(1+idx, folder_name), 'w', encoding='utf-8') as w:
        # sorting the dictionary in descending order
        sorted_scores = {k: v for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)}
        # looping over the sorted scores and writing to file
        for k, v in sorted_scores.items():
            w.write('{} {}\n'.format(k, v if v > 0 else 0))
    return True


def write_tfidf_files(score: dict, idx: int, folder_name: str) -> dict:
    """
    this writes the tf idf scores to files in the folder TF-IDF which is in the root folder
    """
    # create the TF-IDF folder if does not exist
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    topic_num = "R{}".format(101+idx)
    # looping over the 50 queries and writing score files one by one
    with open('{1}/{0}.txt'.format(topic_num, folder_name), 'w', encoding='utf-8') as w:
        sorted_scores = {k: v for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)}
        for k, v in sorted_scores.items():
            w.write('{0} {1} {2}\n'.format(topic_num, k, 1 if v != 0 else 0))
    return sorted_scores


def tf_idf_model(queries: dict, data: dict) -> dict:
    """
    this function uses the tfidf model to find the scores of the query in different documents, assigning a score to the
    document in one of the training_set folders, the score is calculated in such a way:
    normalized term frequency * normalized inverse document frequency
    """
    # tokenizing the query words
    query_list = filtering_stemming(queries.title.split())

    df = data.calc_df()
    all_scores = {}

    # for key, query in query_list.items():
    scores = {}
    for doc_key, pair_dc in data.training_data.items():
        tf_idf = 0
        for q in query_list:
            # (0, 0) incase the query word q does not exist in the training_data corpus
            pair = pair_dc.get(q, False)
            if pair is not False:
                # calculating the tf idf score
                tf_idf += pair[1] * log(data.get_total_docs()/df.get(q, 0))
                break
        scores[doc_key] = tf_idf
        # all_scores[key] = scores

    return scores


def dirichlet_model(queries: dict, data: dict) -> dict:
    """
    this function uses the dirichlet model to find the scores of the query in different documents, assigning a score
    to the document in one of the training_set folders,
    This model is used to find scores of documents which have a small number of terms since it adds a background score
    to the overall score of the query in the document. This model has a tendency to average well over all the corpus
    documents, it uses the following formula;
    dirichlet_score = N/N+u (the probablity of the term to be in document X) + u/u+N (the probablity of the term to
    be in the entire corpus of documents)
    """
    q_list = filtering_stemming(queries.title.split())

    # constant is the average document length
    mu = data.get_avg_doc_len()
    df = data.calc_df()

    scores = {}
    for doc_key, value in data.training_data.items():
        temp = 0
        for q in q_list:
            score_foreground = data.total_words[doc_key] / (mu + data.total_words[doc_key])
            if data.total_words[doc_key] != 0:
                # probability of the query word in the document
                score_foreground *= (df.get(q, 0) / data.total_words[doc_key])
            else:
                score_foreground = 0

            score_background = mu / (mu + data.total_words[doc_key])
            if q in value.keys():
                # probability of the query word in the corpus
                score_background *= df.get(q, 0) / data.get_total_corpus_words()
            else:
                score_background = 0
            temp += score_foreground + score_background

        scores[doc_key] = temp

    return scores


def write_relevance_dat_files(relevance_dc: dict, topic: str, folder_name: str) -> bool:
    """
    this writes the information filtering rocchio model scores to files in the folder IF-ROCCHIO-MODEL which is in
    the root folder
    """
    # creating the folder if it does not exist
    folder = folder_name
    if not os.path.exists(folder):
        os.mkdir(folder)

    with open('{0}/{2}{1}.dat'.format(folder, topic+101, folder_name), 'w', encoding='utf-8') as w:
        for score in relevance_dc.items():
            w.write('{0} {1}\n'.format(score[0], score[1]))

    return True


def rocchio_information_filtering(data: dict, tf_counts: dict, query: dict):
    """
    This uses the Roccio information filtering model to calculate the scores of the different queries in the documents
    in the corpus of training sets
    :param data: relevant documents
    :param tf_counts: the term frequency of different terms in documents
    :param query: the list of query words
    :return:
    """
    query_df = tf_counts.get_new_vector()
    idf_scores = tf_counts.calc_df()

    alpha = 2
    beta = 4
    gamma = 1

    # creating query vector
    for q in query:
        query_df[q] = query.count(q) * alpha

    for key, value in tf_counts.training_data.items():
        # relevant document
        if data[key] == 1:
            for val in value.values():
                d = (1+log(val[1]) if val[1] > 0 else 0) * log(tf_counts.get_total_docs()/idf_scores.get(val[0], 0))
                query_df[val[0]] += d*(beta/list(data.values()).count(1))

        # non relevant doc
        if data[key] == 0:
            for val in value.values():
                d = (1+log(val[1]) if val[1] > 0 else 0) * log(tf_counts.get_total_docs() / idf_scores.get(val[0], 0))
                query_df[val[0]] -= d * (gamma/list(data.values()).count(0))

    # these are the features
    features = [key for key, val in query_df.items() if val > 0]

    relevance_dic = {}
    # finding the relevance scores in the different documents in one of the 50 training sets
    for key, val in tf_counts.training_data.items():
        for _f in features:
            relevance_dic[key] = query_df[_f] * (1 if _f in val.keys() else 0) + relevance_dic.get(key, 0)

    # sorting the scores in the relevance dictionary based on values
    relevance_dic = {k: v for k, v in sorted(relevance_dic.items(), key=lambda item: item[1], reverse=True)}
    return relevance_dic


def probability_based_information_filtering(data: dict, tf_counts: dict, query: dict):
    """
    this function returns the probability based model scores i.e. w(4) {in slide of information filtering}, which
    is one of the best models in information filtering, according to many.
    """

    # number of relevant documents which contain the query word
    relevant_docs = [key for key, val in data.items() if val != 0]

    N = tf_counts.get_total_docs()
    R = len(relevant_docs)
    dc_n = tf_counts.calc_df()
    # number of documents which contain the query word
    # dc_n = {q: tf_counts.get_total_docs_containing_term(q) for q in query}

    dc_r = {q: tf_counts.get_count_in_relevant_docs(q, relevant_docs) for q in dc_n.keys()}

    # calculating the scores of terms in the training set, as a vector of scores
    _temp_n = {k: i+0.5 for k, i in dc_r.items()}
    _temp_d = {k: R+i+0.5 for k, i in dc_r.items()}
    weight_vector_n = {k: i/_temp_d[k] for k, i in _temp_n.items()}

    _temp_n = {k: dc_n[k]-i+0.5 for k, i in dc_r.items()}
    _temp_d = {k: (N-dc_n[k])-(R-i)+0.5 for k, i in dc_r.items()}
    weight_vector_d = {k: i / _temp_d[k] for k, i in _temp_n.items()}

    # the final vector weight score
    weight_vector = {k: i / weight_vector_d[k] for k, i in weight_vector_n.items()}

    relevance_dic = {}
    # ranking the scores of the queries in different documents
    for key, doc in tf_counts.training_data.items():
        _sum = 0
        for term in doc.keys():
            if term in weight_vector.keys():
                _sum += weight_vector[term]
        relevance_dic[key] = _sum

    # sorting the relevance dictionary based on values
    relevance_dic = {k: v for k, v in sorted(relevance_dic.items(), key=lambda item: item[1], reverse=True)}
    return relevance_dic


def evaluate_model(if_file: str, relevance_judge: str) -> bool:
    """
    this calculates the precision, recall and f1 score of the models, model can be passed in as a parameter 'if_file'
    """
    if_scores = {}
    with open(if_file, 'r') as r:
        data = r.read().split()
        for i, _data in enumerate(data):
            if i % 2 == 0:
                try:
                    if_scores[data[i]] = data[i+1]
                except IndexError:
                    break

    relevance_dic = {}
    with open(relevance_judge, 'r') as r:
        data = r.read().split('\n')
        data = filter(lambda x: x != "", data)
        for _data in data:
            _data = _data.split(' ')
            relevance_dic[_data[1]] = _data[2]

    total_relevant = [k for k, v in relevance_dic.items() if v == '1']
    tp = 0
    precision_at_k = 10
    for i, if_score in zip(range(precision_at_k), list(if_scores.keys())[:10]):
        if if_score in total_relevant:
            tp += 1

    # finding top k precision, recall and f1 score
    top_k_precision = tp/precision_at_k
    recall = tp/len(total_relevant)
    if top_k_precision + recall == 0:
        f1_score = 0
    else:
        f1_score = (2 * top_k_precision * recall) / (top_k_precision + recall)

    return top_k_precision, recall, f1_score


if __name__ == '__main__':

    stop_word_list = get_stop_words()
    dataset = []

    path = "Tasks2/Tasks2/dataset101-150/*"
    files = glob(path)

    """all document"""
    for file in files:
        dataset.append(loads_dataset(file))

    """one document"""
    # R107
    # dataset.append(loads_dataset(files[0]))
    topic_defs = {}

    with open("Tasks2/Tasks2/Topic_definitions.txt") as read:
        title, doc_num = '', ''
        for r in read.readlines():

            if r.startswith('<num> Number: '):
                doc_num = r.replace('<num> Number: ', '').rstrip()
            if r.startswith('<title>'):
                title = r.replace('<title>', '').rstrip()
            if title:
                topic_defs[doc_num] = Topics(_title=title)

    # Q2, algo 1, tf idf model
    ls_tf_idf_score = {}
    for idx in range(50):
        doc_num = "R{}".format(101+idx)
        tf_idf_scores = tf_idf_model(topic_defs[doc_num], get_set(dataset, 101+idx))
        dc = write_tfidf_files(tf_idf_scores, idx, 'TF-IDF')
        ls_tf_idf_score[doc_num] = dc

    # Q2, algo 2, Dirichlet smoothing
    ls_dirichlet_score = {}
    for idx in range(50):
        doc_num = "R{}".format(101+idx)
        dirichlet_scores = dirichlet_model(topic_defs[doc_num], get_set(dataset, 101+idx))
        dc = write_dat_files(dirichlet_scores, idx, 'DIRICHLET')
        ls_dirichlet_score[doc_num] = dc

    # tokenizing the query words
    query_list = {}
    for k, v in topic_defs.items():
        query_list[k] = filtering_stemming(v.title.split())

    # Q5, algo 1
    for idx in range(50):
        doc_num = "R{}".format(101+idx)
        relevance_model_1 = rocchio_information_filtering(ls_tf_idf_score[doc_num], get_set(dataset, 101+idx),
                                                          query_list[doc_num])
        write_relevance_dat_files(relevance_model_1, idx, 'IF-ROCCHIO-MODEL')

    # Q5, algo 2
    for idx in range(50):
        doc_num = "R{}".format(101+idx)
        relevance_model_2 = probability_based_information_filtering(ls_tf_idf_score[doc_num],
                                                                    get_set(dataset, 101+idx),
                                                                    query_list[doc_num])
        write_relevance_dat_files(relevance_model_2, idx, 'PROBABALISTIC_MODEL')

    # Q6 - INFORMATION FILTERING MODEL SCORES ARE BETTER
    if_model_precision_ls = []
    if_model_recall_ls = []
    if_model_f1score_ls = []
    with open('EResult1.dat', 'w', encoding='utf-8') as w:
        w.write('Topic  precision  recall     F1\n')
        for idx in range(101, 151, 1):
            t, r, f, = evaluate_model(if_file='IF-ROCCHIO-MODEL/IF_Result{0}.dat'.format(idx),
                                      relevance_judge='Tasks2/Tasks2/Relevance_judgments/Training{0}.txt'.format(idx))
            p = round(t, 4)
            if_model_precision_ls.append(p)
            r = round(r, 4)
            if_model_recall_ls.append(r)
            f = round(f, 4)
            if_model_f1score_ls.append(f)

            w.write(f'{idx:<6} {p:<10} {r:<10} {f}\n')

    # Q2, algo 2
    # dirichlet_scores = dirichlet_model(topic_defs, dataset[0])

    # Q3 - BM25 MODEL SCORES
    ls_bm25_scores = {}
    for idx in range(50):
        doc_num = "R{}".format(101+idx)
        dc = bm25_baseline_model(query_list=query_list[doc_num], data=get_set(dataset, 101+idx))
        write_dat_files(score=dc, idx=idx, folder_name='BM25')
        ls_bm25_scores[doc_num] = dc

    base_model_precision_ls = []
    base_model_recall_ls = []
    base_model_f1score_ls = []
    with open('EResult2.dat', 'w', encoding='utf-8') as w:
        w.write('Topic  precision  recall     F1\n')
        for idx in range(101, 151, 1):
            t, r, f, = evaluate_model(if_file='BM25/B_Result{0}.dat'.format(idx-100),
                                      relevance_judge='Tasks2/Tasks2/Relevance_judgments/Training{0}.txt'.format(idx))
            p = round(t, 4)
            base_model_precision_ls.append(p)
            r = round(r, 4)
            base_model_recall_ls.append(r)
            f = round(f, 4)
            base_model_f1score_ls.append(f)
            w.write(f'{idx:<6} {p:<10} {r:<10} {f}\n')

    # conducting the t test
    np_arr1 = np.array(if_model_precision_ls)
    np_arr2 = np.array(base_model_precision_ls)

    t_test = stats.ttest_ind(a=np_arr1, b=np_arr2, equal_var=True)
    print(t_test)
    print("Since the pvalue of our precision scores is {0}, which is higher than 0.05, we can reject the null "
          "hypothesis 'there is no difference in effectiveness between your IF Model and the baseline model'"
          .format(t_test.pvalue))

"""
EResult1.dat is for the IF-ROCCHIO model
EResult2.dat is for the basline bm25 model
"""
