import numpy as np
import xlrd
 
 
def load_data_and_label(path):
    data = xlrd.open_workbook(path).sheets()[0]
    docs = []
    labels = []
    for i in range(data.nrows):
        row_values = data.row_values(i)
        doc = row_values[0].split()
        label = row_values[1]
        docs.append(doc)
        labels.append(label)
    labels = np.array(labels)
    labels = labels.astype('int32')
    return [docs, labels]
 
def pad_sentences(sentences, max_len):
    padd_sentences = []
    for sentence in sentences:
        if len(sentence) < max_len:
            num_padding = max_len - len(sentence)
            new_sentence = sentence + ["</s>"] * num_padding
        elif len(sentence) > max_len:
            new_sentence = sentence[:max_len]
        else:
            new_sentence = sentence
        padd_sentences.append(new_sentence)
    return padd_sentences
 
def load_pretrained_word2vec(infile):
    if isinstance(infile, str):
        infile = open(infile)
    word2vec = {}
    vec_len = 0
    for idx, line in enumerate(infile):
        tks = line.strip().split()
        if idx==0:
            vec_len = len(tks[1:])
        word2vec[tks[0]] = list(map(float, tks[1:])) 
 
    if "</s>" in word2vec:
        word2vec["</s>"] = [0]*vec_len
    return word2vec
 
def build_input_with_word2vec(sentences, labels, word2vec):
    vecs = []
    for sentence in sentences:
        vec = []
        for word in sentence:
            if word in word2vec:
                vec.append(word2vec[word])
            else:
                vec.append(word2vec["</s>"])
        vecs.append(vec)
    vecs = np.array(vecs)
    labels = np.array(labels)
    return [vecs, labels]
 
def pad_data_with_batchsize(data, batch_size):
    num_pad = batch_size - data.shape[0] % batch_size
    data = np.concatenate((data, data[:num_pad]), axis=0)
    return data
 
 
def load_data_with_word2vec(data_path, word2vec_path, max_len):
    sentences, labels = load_data_and_label(data_path)
    sentences = pad_sentences(sentences, max_len)
    word2vec = load_pretrained_word2vec(word2vec_path)
    return build_input_with_word2vec(sentences, labels, word2vec)
 
 
if __name__ == '__main__':
    sentences, labels = load_data_with_word2vec("D:\\train1.xlsx", "D:\\en_word_vec.txt", 128)
    print(sentences.shape)
    print(labels.shape)