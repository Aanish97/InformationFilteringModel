import errno
import string
from glob import glob
from porter2stemmer import Porter2Stemmer


def get_stop_words():
    path_stop_list = "common-english-words.txt"
    reader = open(path_stop_list, "r")
    stop_list = reader.read()
    reader.close()
    return stop_list.split(',')


def parse_doc(file):
    # myfile = open('inputdata/807606newsML.xml')
    myfile = open(file)

    curr_doc = {}
    start_end = False

    file_ = myfile.readlines()
    word_count = 0
    for line in file_:
        line = line.strip()
        if start_end == False:
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
    myfile.close()

    new_dic = {}
    for i in curr_doc.keys():
        new_dic[i] = [i, curr_doc[i]]

    return new_dic, (docid, word_count)


if __name__ == '__main__':

    stop_word_list = get_stop_words()

    path = "Tasks2/Tasks2/dataset101-150/*"
    files = glob(path)

    for name in files:
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




