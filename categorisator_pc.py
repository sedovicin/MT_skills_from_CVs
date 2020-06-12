import keras
from keras.utils import Sequence
import nn_input_data_mgmt as nn_mgmt
import dataset_corpus_mgmt as dc_mgmt
import numpy as np
from keras.preprocessing.sequence import pad_sequences

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

import vectorizators
word2vec = vectorizators.get_word2vec(word2vec_path='word2vec.obj', corpus_path='corpus_train.json')
def get_batch(index):

	categories = dict()

	sentences_skills = dc_mgmt.file_to_tokens('cv_extracted/skills/%s_skills.txt' % index)
	for sentence in sentences_skills:
		for word in sentence:
			categories[word] = 1
	cv_phrases = nn_mgmt.get_phrases('cv_extracted/cvs/%s_cv.txt' % index, categories)
	y = list()
	pre_list = list()
	phr_list = list()
	post_list = list()
	i = 0
	for sent in cv_phrases:
		for phrase in sent:
			pre = list()
			for word in phrase.pre_phrase_context:
				pre.append(word2vec.wv.get_vector(word))
			phr = list()
			for word in phrase.phrase:
				phr.append(word2vec.wv.get_vector(word))
			post = list()
			for word in phrase.post_phrase_context:
				post.append(word2vec.wv.get_vector(word))
			if len(pre) == 0:
				pre = np.zeros((1, word2vec.wv.vector_size)).tolist()
			# else:
			# 	pre = np.array(pre)
			# phr = np.array(phr)
			if len(post) == 0:
				post = np.zeros((1, word2vec.wv.vector_size)).tolist()
			# else:
			# 	post = np.array(post)

			# print(np.shape(pre))
			# print(np.shape(phr))
			# print(np.shape(post))
			# print("")
			# pre = [np.array(pre)]
			# phr = [np.array(phr)]
			# post = [np.array(post)]
			# all_together = [pre, phr, post]

			pre_list.append(pre)
			phr_list.append(phr)
			post_list.append(post)

			# print(np.shape(pre))
			# print(np.shape(phr))
			# print(np.shape(post))
			# print("")
			# x.append(all_together)
			y_entry = np.zeros(2)

			y_entry[phrase.category] = 1

			y.append(y_entry)
			i+=1
			# if i == 2:
			# 	return np.array(pre_list), np.array(phr_list), np.array(post_list), np.array(y)
	# print(type(x))
	# print(np.shape(x))
	# print(type(y))
	# print(np.shape(y))
	return \
		pad_sequences(np.array(pre_list), padding='pre'), \
		pad_sequences(np.array(phr_list)),\
		pad_sequences(np.array(post_list), padding='post'), \
		np.array(y)


from keras.layers import Input, Dense, LSTM, concatenate, Masking
from keras.models import Model
import vectorizators

generator = DataGenerator()

model_input1 = Input(shape=(None, generator.word2vec.vector_size))
model_input2 = Input(shape=(None, generator.word2vec.vector_size))
model_input3 = Input(shape=(None, generator.word2vec.vector_size))

masking1 = Masking()(model_input1)
masking2 = Masking()(model_input2)
masking3 = Masking()(model_input3)

lstm1 = LSTM(256)(masking1)
lstm2 = LSTM(256)(masking2)
lstm3 = LSTM(256)(masking3)

dense1 = Dense(128, activation='relu')(lstm1)
dense2 = Dense(128, activation='relu')(lstm2)
dense3 = Dense(128, activation='relu')(lstm3)

x = concatenate([dense1, dense2, dense3])

dense = Dense(64, activation='relu')(x)
dense = Dense(16, activation='relu')(dense)
main_output = Dense(2, activation='softmax')(dense)

model = Model(inputs=[model_input1, model_input2, model_input3], outputs=main_output)
# model = Model(inputs=model_input1, outputs=main_output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

for i in range(1, 100):
	pr, ph, po, y = get_batch(i)
	# print(np.shape(pr))
	# print(np.shape(ph))
	# print(np.shape(po))
	# print(np.shape(y))
	# print("")
	print(i)
	model.train_on_batch(x=[pr, ph, po], y=y, reset_metrics=False)
# model.fit_generator(generator)
pr, ph, po, y = get_batch(101)
print(model.evaluate(x=[pr, ph, po], y=y))