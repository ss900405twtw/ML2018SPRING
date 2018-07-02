import os,sys,re,jieba
import numpy as np
from gensim.models import word2vec
from keras.models import load_model,Model
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM,Embedding,Input,multiply,merge,GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint

# def listdir_fullpath(d):
#     train = np.zeros((len(os.listdir(d)),),dtype='object')
#     return [ os.path.join(d, f) for f in os.listdir(d) ] , train
def sentence_preprocessing(sentence,mode):
    if mode == 'train':
        sentence = re.sub('[a-zA-Z0-9]|[，..."]', "" , sentence)
        return sentence
    else:
        id_q_a = sentence.split(",")
        id_q_a[1] = re.sub(r'\s', "", id_q_a[1])
        id_q_a[2] = re.sub(r'([0-9]:)', "", id_q_a[2]).strip().split("\t")
        return np.asarray([id_q_a[1]] + id_q_a[2])
def get_data_list(filepath,mode):
# 資料格式
    if mode == 'train':
        train_dir , train_data = listdir_fullpath(filepath)
        for i,directory in enumerate(train_dir):
            list_sentence = []
            with open(directory , 'r', encoding='utf8') as file:
                for line in file:
                    line = line.strip('\n')
                    line = sentence_preprocessing(line,'train')
                    list_sentence.append(line)
                file.close()
            train_data[i] = list_sentence
        return train_data
    else:
        path = os.path.join(filepath, 'testing_data.csv')
        test_data = np.empty((5060, 7), dtype='object')
        with open(path, 'r', encoding='utf8') as file:
            first_line = file.readline()
            for i, line in enumerate(file):
                test_data[i, :] = sentence_preprocessing(line,'test')
            file.close()
        return test_data
def jieba_separate_word(data,mode):
# 測試要用jieba何種模式、HMM與否、轉簡體與否、set_deictionary與否....等
    from opencc.opencc import OpenCC
    translation = OpenCC('tw2s')
    jieba.set_dictionary('dict.txt.big')
    if mode == 'train':
        words = np.zeros((data.shape[0],),dtype='object')
        for i,list in enumerate(data):
            words[i] = []
            for sentence in list:
                words[i].append(jieba.lcut(translation.convert(sentence),cut_all=False))
        return words
    else:
        words = np.empty((5060,7),dtype='object')
        for i in range(words.shape[0]):
            for j in range(words.shape[1]):
                words[i,j] = jieba.lcut(translation.convert(data[i,j]),cut_all=False)
        return words
def get_index_array(words_list,max_length):
    index_array = np.zeros((len(words_list),max_length),dtype='int32')
    for index in range(len(words_list)):
        for i, word in enumerate(words_list[index]):
            try:
                index_array[index,i] = word_dict.wv.vocab[word].index + 1
            except:
                pass
    return index_array


testing_data = get_data_list('./dataset','test')
testing_sep_data = jieba_separate_word(testing_data,'test')
word_dict = word2vec.Word2Vec.load("dim40.bin")

model = load_model("./0628_dualencoder_test_yuchi.h5")
answer = np.zeros((5060,),dtype='int32')
for i in range(5060):
    testing_sequence = get_index_array(testing_sep_data[i],15)
    test_q_sequence = np.tile(testing_sequence[0,:],(6,1))
    test_a_sequence = testing_sequence[1:7,:]
    predict = model.predict([test_q_sequence,test_a_sequence])
    answer[i] = np.argmax(predict,axis=0)[0]
f = open('0628_dualencoder_test_yuchi.csv','w')
f.write('id,ans\n')
for i,value in enumerate(answer):
    f.write("%d,%s\n" % (i , value))
f.close()


model = load_model("./0627_dualencoder_test_yuchi.h5")
answer = np.zeros((5060,),dtype='int32')
for i in range(5060):
    testing_sequence = get_index_array(testing_sep_data[i],15)
    test_q_sequence = np.tile(testing_sequence[0,:],(6,1))
    test_a_sequence = testing_sequence[1:7,:]
    predict = model.predict([test_q_sequence,test_a_sequence])
    answer[i] = np.argmax(predict,axis=0)[0]
f = open('0627_dualencoder_test_yuchi.csv','w')
f.write('id,ans\n')
for i,value in enumerate(answer):
    f.write("%d,%s\n" % (i , value))
f.close()

word_dict = word2vec.Word2Vec.load("0612_test1_yuchi_0.47905.model")
model = load_model("./0614_dualencoder_test1_yuchi.h5")
answer = np.zeros((5060,),dtype='int32')
for i in range(5060):
    testing_sequence = get_index_array(testing_sep_data[i],15)
    test_q_sequence = np.tile(testing_sequence[0,:],(6,1))
    test_a_sequence = testing_sequence[1:7,:]
    predict = model.predict([test_q_sequence,test_a_sequence])
    answer[i] = np.argmax(predict,axis=0)[0]
f = open('0614_dualencoder_test1_yuchi.csv','w')
f.write('id,ans\n')
for i,value in enumerate(answer):
    f.write("%d,%s\n" % (i , value))
f.close()







