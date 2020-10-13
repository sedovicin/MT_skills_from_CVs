from keras.layers import Input, Dense, LSTM, concatenate, Masking
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import nn_input_data_mgmt as nn_mgmt


class PhraseContextCategorisator:
	def __init__(self, vector_size, model=None):
		self.vector_size = vector_size
		if model is None:
			self.model = self.__create_model()
		else:
			self.model = load_model(model)
		print(self.model.summary())
		self.generator = nn_mgmt.SampleGenerator(context_size=3, categories_count=2)

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
		for i in range(1, 10001, batch_size):
			print("Training from %s to %s..." % (i, i+batch_size))
			pre, phr, post, y = self.generator.get_batch_x_y(i, i+batch_size)
			phr = pad_sequences(phr, dtype='float32')
			self.model.train_on_batch(x=[pre, phr, post], y=y)
		self.model.save('cv_model')

	def evaluate(self, batch_size):

		for i in range(10001, 10594, batch_size):
			print("Evaluating from %s to %s..." % (i, i + batch_size))
			pre, phr, post, y, pre_word, phr_word, post_word = self.generator.get_batch_x_y(i, i + batch_size)
			phr = pad_sequences(phr, dtype='float32')
			print(self.model.evaluate(x=[pre, phr, post], y=y, verbose=0))

	def predict(self, mark):
		print("Predicting %s..." % mark)
		pre, phr, post, y, pre_word, phr_word, post_word = self.generator.get_batch_x_y(mark, mark+1)
		phr = pad_sequences(phr, dtype='float32')
		result = self.model.predict(x=[pre, phr, post], verbose=0)
		return pre_word, phr_word, post_word, y, result


def main():
	import numpy as np
	categorisator = PhraseContextCategorisator(1)
	categorisator.evaluate(1)

	pre, phr, post, y, result = categorisator.predict(54)
	for i in range(len(pre)):
		print("%s ; %s ; %s ; %s ; %s" % (pre[i], phr[i], post[i], y[i], result[i]))

	for line in result:
		line[0] = int(round(line[0]))
		line[1] = int(round(line[1]))

	y = np.array(y)
	y = y[:, 1]
	result = np.array(result)
	result = result[:, 1]
	print(accuracy_score(y, result))
	print(precision_score(y, result))
	print(recall_score(y, result))
	print(f1_score(y, result))
	print("UKUPNO %s PRIMJERA" %len(y))


if __name__ == "__main__":
	main()
