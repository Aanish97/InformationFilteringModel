import copy
import errno
import os
import string
from math import log
from glob import glob
from porter2stemmer import Porter2Stemmer


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
    this would have docs of 1 complete Training set e.g. Training105
    """

    def __init__(self, training_set_name):
        self.training_data = {}
        self.total_words = {}
        self.name = training_set_name

    def add_item(self, training_data, doc_id, total_words):
        self.training_data[doc_id] = training_data
        self.total_words[doc_id] = total_words

    def get_avg_doc_len(self):
        return round(sum(self.total_words.values())/len(self.total_words.keys()))

    def get_total_docs(self):
        return len(self.total_words.keys())

    def calc_df(self):
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


def get_stop_words() -> list:
    """
    :return: list of all the stop words in the folder
    """
    path_stop_list = "common-english-words.txt"
    reader = open(path_stop_list, "r")
    stop_list = reader.read()
    reader.close()
    return stop_list.split(',')


def parse_doc(file: str) -> list:
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


def filtering_stemming(t):
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


def bm25_baseline_model(queries: dict, data: dict):
    # qs = [
    #     ['British', 'fashion'],
    #     ['fashion', 'awards'],
    #     ['Stock', 'market'],
    #     ['British', 'Fashion', 'Awards']
    # ]

    # tokenizing the query words
    query_list = {}
    for k, v in queries.items():
        query_list[k] = filtering_stemming(v.title.split())

    K = 0
    k1 = 1.2
    k2 = 100  # can be changed btw 0 to 1000
    b = 0.75
    sample_list = [0] * data.get_total_docs()
    scores = {}
    mu = data.get_avg_doc_len()

    df = data.calc_df()

    for key, query in query_list.items():
        flag1 = 0
        scores_temp0 = {}
        for q in query:
            scores_temp = {}
            for i, p in zip(range(len(files)), data.training_data.keys()):
                temp0 = data.total_words[p] / mu
                temp0 = b * temp0
                temp0 = (1 - b) + temp0
                K = k1 * temp0

                # numerator
                temp_var3 = (1 + k2) * query.count(q)
                # num/denom
                temp_var3 = temp_var3 / (k2 + query.count(q))

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

        scores[key] = scores_temp0

    return scores


def write_dat_files(scores: dict) -> bool:
    if not os.path.exists('BM25'):
        os.mkdir('BM25')
    for c, score in zip(range(1, 51, 1), scores.keys()):
        with open('BM25/B_Result{0}.dat'.format(c), 'w', encoding='utf-8') as w:
            sorted_scores = {k: v for k, v in sorted(scores[score].items(), key=lambda item: item[1], reverse=True)}
            for k, v in sorted_scores.items():
                w.write('{} {}\n'.format(k, v))
    return True


if __name__ == '__main__':

    stop_word_list = get_stop_words()
    dataset = []

    path = "Tasks2/Tasks2/dataset101-150/*"
    files = glob(path)

    """all document"""
    # for file in files:
    #     dataset.append(loads_dataset(file))

    """one document"""
    dataset.append(loads_dataset(files[1]))
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

    bm25_scores = bm25_baseline_model(topic_defs, dataset[0])

    write_dat_files(bm25_scores)






