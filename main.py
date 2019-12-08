import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import pickle
from datetime import datetime


class ChatBot:

    inputs = []
    responses = []

    def prep_data(self, data):
        stemmer = LancasterStemmer()

        words = []
        labels = []
        docs_x = []
        docs_y = []

        for intent in data['intents']:
            for pattern in intent['patterns']:
                words_in_pattern = nltk.word_tokenize(pattern)
                words.extend(words_in_pattern)
                docs_x.append(words_in_pattern)
                docs_y.append(intent['tag'])

            if intent['tag'] not in labels:
                labels.append(intent['tag'])

        words = [stemmer.stem(w.lower()) for w in words if w != '?']
        words = sorted(list(set(words))) # set to remove duplicates

        labels = sorted(labels)

        train = []
        output = []

        out_empty = [0 for _ in range(len(labels))]

        # understand this better
        for x, doc in enumerate(docs_x):
            bag = []
            
            wrds = [stemmer.stem(w.lower()) for w in doc]
            # this isn't right
            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    # print(w)
                    bag.append(0)

            output_row = out_empty[:] # copy of out_empty
            output_row[labels.index(docs_y[x])] = 1

            train.append(bag)
            output.append(output_row)

        train = np.array(train)
        output = np.array(output)

        with open('data.pickle', 'wb') as f:
            pickle.dump((words, labels, train, output), f)

    def init_model(self, train, output):
        tf.reset_default_graph()

        net = tflearn.input_data(shape=[None, len(train[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
        net = tflearn.regression(net)

        return tflearn.DNN(net)

    def bag_of_words(self, s, words):
        stemmer = LancasterStemmer()
        bag = [0 for _ in range(len(words))]

        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1

        return np.array(bag)

    def get_bot_response(self, inp, data, labels, model, words):
        result = model.predict([self.bag_of_words(inp, words)])
        result_index = np.argmax(result)
        tag = labels[result_index]

        if result[0][result_index] < 0.7:
            response = 'I didn\'t understand that.'
        else:
            # handle special cases
            if tag == 'time':
                response = datetime.now().strftime('%Y-%m-%d %H:%M')
            else:
                for t in data['intents']:
                    if t['tag'] == tag:
                        responses = t['responses']
                        response = random.choice(responses)
                        break

        self.inputs.insert(0, inp)
        self.responses.insert(0, response)

        return response
