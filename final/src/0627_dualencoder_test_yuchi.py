import os,sys,re,jieba
import numpy as np
from gensim.models import word2vec
from keras.models import load_model,Model
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM,Embedding,Input,multiply,merge,GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint

def listdir_fullpath(d):
    train = np.zeros((len(os.listdir(d)),),dtype='object')
    return [ os.path.join(d, f) for f in os.listdir(d) ] , train
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
def concat_sentence(words,test_words,window_length,include_test):
    all_set_sentence = []
    for x in range(5):
        a_set_sentence = []
        for index in range(len(words[x]) - window_length):
            sentence = []
            for i in range(window_length):
                sentence.extend(words[x][index+i])
            a_set_sentence.append(sentence)
        all_set_sentence.extend(a_set_sentence)
    if include_test == True:
        sentence = []
        for i in range(7):
            for j in range(5060):
                sentence.append(test_words[j,i])
        all_set_sentence.extend(sentence)
    return all_set_sentence
def create_word2vec_dict(words,dim):
    all_words = []
    for x in range(words.shape[0]):
        all_words.extend(words[x])
    dictionary = word2vec.Word2Vec(all_words, sg=1, size=dim, min_count=1, workers=4)
    return dictionary
def get_max_length(words):
    max_length = 0
    for x in range(5):
        for i in range(len(words[x])):
            if(len(words[x][i])>max_length):
                max_length = len(words[x][i])
    return max_length
def get_index_array(words_list,max_length):
    index_array = np.zeros((len(words_list),max_length),dtype='int32')
    for index in range(len(words_list)):
        for i, word in enumerate(words_list[index]):
            try:
                index_array[index,i] = word_dict.wv.vocab[word].index + 1
            except:
                pass
    return index_array
def create_embedding_matrix(dictionary,dim):
    embedding_matrix = np.zeros((len(dictionary.wv.vocab) + 1, dim), dtype='float')
    for word, vocab_object in dictionary.wv.vocab.items():
        # if vocab_object.index==0:
        #     print(word,vocab_object.index)
        embedding_matrix[vocab_object.index + 1] = dictionary[word]
    return embedding_matrix
def dual_encoder(sequence_length,embedding_matrix,input_dim):
    model = Sequential()
    model.add(Embedding(input_dim,embedding_matrix.shape[1],input_length=sequence_length,weights=[embedding_matrix],trainable=True))
    # model.add(LSTM(units=300, activation='tanh',dropout=0.3,recurrent_dropout=0.3, return_sequences=True))
    model.add(LSTM(units=300, activation='tanh', return_sequences=True))
    model.add(GRU(units=300, activation='tanh', return_sequences=False))
    context_input = Input(shape=(sequence_length,), dtype='int32')
    response_input = Input(shape=(sequence_length,), dtype='int32')
    context_branch = model(context_input)
    response_branch = model(response_input)
    # concatenated = Merge([context_branch, response_branch], mode='mul')
    concatenated = multiply([context_branch, response_branch])
    # dnnout = Dense(10, activation="sigmoid")(concatenated)
    out = Dense(1, activation="sigmoid")(concatenated)
    dual_encoder = Model([context_input, response_input], out)
    dual_encoder.compile(loss='binary_crossentropy',optimizer='adam', metrics=["accuracy"])
    return dual_encoder
def split_data(a,b,c,size,ratio):
    X = a
    Y = b
    Z = c
    val_size = int(size * ratio)
    return X[val_size:], Y[val_size:], Z[val_size:], X[:val_size], Y[:val_size], Z[:val_size]
def get_totalwords_averagelength(words):
    total = 0
    for i in range(len(words)):
        total = total + len(words[i])
    average = total / len(words)
    return total,average
def combine_index_array(words_list,max_length):
    index_array = np.zeros((len(words_list),max_length),dtype='int32')
    answer_array = np.zeros((len(words_list), max_length), dtype='int32')
    for index in range(len(words_list)):
        current = max_length
        a = index
        while (current > len(words_list[a]) and a < len(words_list)-1):
            for i, word in enumerate(words_list[a]):
                try:
                    index_array[index,max_length-current+i] = word_dict.wv.vocab[word].index + 1
                except:
                    pass
            current = current - len(words_list[a])
            a = a + 1
        for i, word in enumerate(words_list[a]):
            try:
                answer_array[index, i] = word_dict.wv.vocab[word].index + 1
            except:
                pass
    return index_array,answer_array

training_data = get_data_list('./dataset/training_data','train')
training_sep_data = jieba_separate_word(training_data,'train')
# testing資料格式(function內有資料預處理函式)和斷詞
testing_data = get_data_list('./dataset','test')
testing_sep_data = jieba_separate_word(testing_data,'test')
dim = 40
# word_dict = create_word2vec_dict(training_words,dim)
word_dict = word2vec.Word2Vec.load("dim40.bin")
# max_length = get_max_length(training_words)
training_sequence_1 = get_index_array(training_sep_data[0],15)     #17000sentences,73166words,average4.3038
training_sequence_2 = get_index_array(training_sep_data[1],15)     #24000sentences,91441words,average3.8100
training_sequence_3 = get_index_array(training_sep_data[2],15)     #34736sentences,163728words,average4.71349
training_sequence_4 = get_index_array(training_sep_data[3],15)     #600000sentences,2859114words,average4.7651
training_sequence_5 = get_index_array(training_sep_data[4],15)     #72408sentences,333305words,average4.60315
# a,b = get_totalwords_averagelength(training_words[4])
# combine_q_sequence_1,combine_a_sequence_1 = combine_index_array(training_words[0],15)
# combine_q_sequence_2,combine_a_sequence_2 = combine_index_array(training_words[1],15)
# combine_q_sequence_3,combine_a_sequence_3 = combine_index_array(training_words[2],15)
# combine_q_sequence_4,combine_a_sequence_4 = combine_index_array(training_words[3],15)
# combine_q_sequence_5,combine_a_sequence_5 = combine_index_array(training_words[4],15)

question_sequence = training_sequence_1[0:16999]
question_sequence = np.concatenate((question_sequence,training_sequence_2[0:23999],training_sequence_3[0:34735],training_sequence_4[0:599999],training_sequence_5[0:72407]))
# question_sequence = np.concatenate((question_sequence,combine_q_sequence_1[0:16900],combine_q_sequence_2[0:23900],combine_q_sequence_3[0:34700],combine_q_sequence_4[0:599900],combine_q_sequence_5[0:72200]))
answer_sequence = training_sequence_1[1:17000]
answer_sequence = np.concatenate((answer_sequence,training_sequence_2[1:24000],training_sequence_3[1:34736],training_sequence_4[1:600000],training_sequence_5[1:72408]))
# answer_sequence = np.concatenate((answer_sequence,combine_a_sequence_1[0:16900],combine_a_sequence_2[0:23900],combine_a_sequence_3[0:34700],combine_a_sequence_4[0:599900],combine_a_sequence_5[0:72200]))
label = np.ones((question_sequence.shape[0],1),dtype='int32')

error_question_sequence = training_sequence_1[0:10000]
error_question_sequence = np.concatenate((error_question_sequence,training_sequence_2[0:20000],training_sequence_3[0:30000],training_sequence_4[0:550000],training_sequence_5[0:70000]))
error_question_sequence = np.tile(error_question_sequence,(6,1))
# error_question_sequence = np.concatenate((error_question_sequence,combine_q_sequence_1[0:16900],combine_q_sequence_2[0:23900],combine_q_sequence_3[0:34700],combine_q_sequence_4[0:599900],combine_q_sequence_5[0:72200]))
error_answer_sequence = training_sequence_1[7000:17000]
error_answer_sequence = np.concatenate((error_answer_sequence,training_sequence_2[3000:23000],training_sequence_3[4000:34000],training_sequence_4[50000:600000],training_sequence_5[2000:72000]))
error_answer_sequence = np.concatenate((error_answer_sequence,training_sequence_1[2000:12000],training_sequence_2[1000:21000],training_sequence_3[2000:32000],training_sequence_4[10000:560000],training_sequence_5[1000:71000]))
error_answer_sequence = np.concatenate((error_answer_sequence,training_sequence_1[3000:13000],training_sequence_2[1500:21500],training_sequence_3[3000:33000],training_sequence_4[20000:570000],training_sequence_5[1500:71500]))
error_answer_sequence = np.concatenate((error_answer_sequence,training_sequence_1[4000:14000],training_sequence_2[2000:22000],training_sequence_3[1000:31000],training_sequence_4[30000:580000],training_sequence_5[500:70500]))
error_answer_sequence = np.concatenate((error_answer_sequence,training_sequence_1[5000:15000],training_sequence_2[2500:22500],training_sequence_3[1500:31500],training_sequence_4[40000:590000],training_sequence_5[800:70800]))
error_answer_sequence = np.concatenate((error_answer_sequence,training_sequence_1[6000:16000],training_sequence_2[500:20500],training_sequence_3[2500:32500],training_sequence_4[45000:595000],training_sequence_5[1300:71300]))
# error_answer_sequence = np.concatenate((error_answer_sequence,training_sequence_1[100:17000],combine_a_sequence_2[50:23950],combine_a_sequence_3[30:34730],combine_a_sequence_4[50:599950],combine_a_sequence_5[200:72400]))
error_label = np.zeros((error_question_sequence.shape[0],1),dtype='int32')

all_correct_data = np.concatenate((question_sequence,answer_sequence,label),axis =1)
all_error_data = np.concatenate((error_question_sequence,error_answer_sequence,error_label),axis =1)
all_data = np.concatenate((all_correct_data,all_error_data),axis =0)
np.random.shuffle(all_data)
# all_data = all_data[0:400000,:]
Q,R,L,Q_val,R_val,L_val = split_data(all_data[:,0:15],all_data[:,15:30],all_data[:,30],all_data.shape[0],0.1)

embedding_matrix = create_embedding_matrix(word_dict,dim)
dual_encoder_model = dual_encoder(15,embedding_matrix,len(word_dict.wv.vocab)+1)
# earlystopping = EarlyStopping(monitor='val_acc', patience=8, verbose=2, mode='max')
# checkpoint = ModelCheckpoint("./earlystop_model.h5", monitor='val_acc', save_best_only=True, verbose=0, mode='max')
earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='min')
checkpoint = ModelCheckpoint("./earlystop_model.h5", monitor='val_loss', save_best_only=True, verbose=0, mode='min')
dual_encoder_model.fit([Q,R],L,validation_data=([Q_val,R_val],L_val), batch_size=1000, epochs=100, verbose=2, callbacks=[checkpoint, earlystopping])
model = load_model("./earlystop_model.h5")


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

# for sentence in training_data[0]:
#     print('/'.join(jieba.cut_for_search(sentence, HMM=False)))