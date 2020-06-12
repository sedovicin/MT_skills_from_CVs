import keras
from keras.utils import Sequence
import nn_input_data_mgmt as nn_mgmt
import dataset_corpus_mgmt as dc_mgmt
import numpy as np
from keras.preprocessing.sequence import pad_sequences

import vectorizators


# word2vec = vectorizators.get_word2vec(word2vec_path='word2vec.obj', corpus_path='corpus_train.json')
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
			if len(post) == 0:
				post = np.zeros((1, word2vec.wv.vector_size)).tolist()

			pre_list.append(pre)
			phr_list.append(phr)
			post_list.append(post)

			y_entry = np.zeros(2)

			y_entry[phrase.category] = 1

			y.append(y_entry)

	return \
		pad_sequences(np.array(pre_list), padding='pre'), \
		pad_sequences(np.array(phr_list)),\
		pad_sequences(np.array(post_list), padding='post'), \
		np.array(y)


from keras.layers import Input, Dense, LSTM, concatenate, Masking
from keras.models import Model




# for i in range(1, 100):
# 	pr, ph, po, y = get_batch(i)
# 	print(i)
# 	model.train_on_batch(x=[pr, ph, po], y=y, reset_metrics=False)
# pr, ph, po, y = get_batch(101)
# print(model.evaluate(x=[pr, ph, po], y=y))

class PhraseContextCategorisator():
	def __init__(self, vector_size):
		self.vector_size = vector_size
		self.model = self.__create_model()
		print(self.model.summary())

	def __create_model(self):
		model_input1 = Input(shape=(None, self.vector_size))
		model_input2 = Input(shape=(None, self.vector_size))
		model_input3 = Input(shape=(None, self.vector_size))

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

		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

		return model

	def train(self, batch_size):
		"""

		:param batch_size: amount of files to be processed in batch
		:return:
		"""
		generator = nn_mgmt.SampleGenerator(corpus_path='corpus_train.json', dataset_path='dataset_train.json')
		for i in range(1, 10001, batch_size):
			print("Training from %s to %s..." %(i, i+batch_size))
			pre, phr, post, y = generator.get_batch_x_y(i, i+batch_size)
			pre = pad_sequences(pre)
			phr = pad_sequences(phr)
			post = pad_sequences(post)
			self.model.train_on_batch(x=[pre, phr, post], y=y)

def main():
	categorisator = PhraseContextCategorisator(100)
	categorisator.train(50)


if __name__ == "__main__":
	main()