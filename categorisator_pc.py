import keras
from keras.utils import Sequence
import nn_input_data_mgmt as nn_mgmt
import dataset_corpus_mgmt as dc_mgmt
import numpy as np

class DataGenerator(Sequence):

	def __init__(self):
		self.categories = dict()

		sentences_skills = dc_mgmt.file_to_tokens('cv_extracted/skills/1_skills.txt')
		for sentence in sentences_skills:
			for word in sentence:
				self.categories[word] = 1
		cv_phrases = nn_mgmt.get_phrases('cv_extracted/cvs/1_cv.txt', self.categories)
		self.cv_phrases = list()
		for sent in cv_phrases:
			for phrase in sent:
				self.cv_phrases.append(phrase)
		self.word2vec = vectorizators.get_word2vec(word2vec_path='word2vec.obj', corpus_path='corpus_train.json')


	def __getitem__(self, index):
		phrase = self.cv_phrases[index].phrase
		pre = self.cv_phrases[index].pre_phrase_context
		post = self.cv_phrases[index].post_phrase_context
		y_cat = self.cv_phrases[index].category

		pre_context = list()
		for word in pre:
			pre_context.append(self.word2vec.wv.get_vector(word))
		# for word in post:
		# 	pre_context.append(self.word2vec.wv.get_vector(word))
		# context.extend(pre)
		# context.extend(post)

		post_context = list()
		for word in post:
			post_context.append(self.word2vec.wv.get_vector(word))

		# all_together.extend(pre)
		# all_together.extend(phrase)
		# all_together.extend(post)

		# print("PHR")
		# print(phrase)
		# print("CONTEXT")
		# print(context)
		# print("ALL")
		# print(all_together)

		phr = list()
		for word in phrase:
			phr.append(self.word2vec.wv.get_vector(word))
		phr = np.array([phr])
		# phr = phr.reshape(phr.shape[0], 1, phr.shape[1])
		if len(pre_context) == 0:
			pre_context = np.zeros((1, 1, self.word2vec.wv.vector_size))
		else:
			pre_context = np.array([pre_context])

		# context = context.reshape(context.shape[0], 1, context.shape[1])
		if len(post_context) == 0:
			post_context = np.zeros((1, 1, self.word2vec.wv.vector_size))
		else:
			post_context = np.array([post_context])


		print(np.shape(pre_context))
		print(np.shape(phr))
		print(np.shape(post_context))

		# phr = keras.preprocessing.sequence.pad_sequences(phr)
		# context = keras.preprocessing.sequence.pad_sequences(context)
		X = [pre_context, phr, post_context]

		y = np.zeros((1,2))
		y[y_cat] = 1

		print(pre)
		print(phrase)
		print(post)
		return X, y

	def __len__(self):
		return 1

from keras.layers import Input, Dense, LSTM, concatenate
from keras.models import Model
import vectorizators

generator = DataGenerator()

model_input1 = Input(shape=(None, generator.word2vec.vector_size))
model_input2 = Input(shape=(None, generator.word2vec.vector_size))
model_input3 = Input(shape=(None, generator.word2vec.vector_size))

lstm1 = LSTM(256)(model_input1)
lstm2 = LSTM(256)(model_input2)
lstm3 = LSTM(256)(model_input3)

dense1 = Dense(128, activation='relu')(lstm1)
dense2 = Dense(128, activation='relu')(lstm2)
dense3 = Dense(128, activation='relu')(lstm3)
x = concatenate([dense1, dense2, dense3])
dense = Dense(64, activation='relu')(x)
dense = Dense(16, activation='relu')(dense)
main_output = Dense(2, activation='softmax')(dense)

model = Model(inputs=[model_input1, model_input2, model_input3], outputs=main_output)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit_generator(generator)