from keras.layers import Input, Dense, LSTM, concatenate, Masking
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

import nn_input_data_mgmt as nn_mgmt


class PhraseContextCategorisator:
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
		for i in range(1, 101, batch_size):
			print("Training from %s to %s..." % (i, i+batch_size))
			pre, phr, post, y = generator.get_batch_x_y(i, i+batch_size)
			pre = pad_sequences(pre, dtype='float32')
			phr = pad_sequences(phr, dtype='float32')
			post = pad_sequences(post, dtype='float32')
			self.model.train_on_batch(x=[pre, phr, post], y=y)
			break

	def evaluate(self, batch_size):
		generator = nn_mgmt.SampleGenerator(corpus_path='corpus_train.json', dataset_path='dataset_train.json')
		for i in range(101, 151, batch_size):
			print("Evaluating from %s to %s..." % (i, i + batch_size))
			pre, phr, post, y = generator.get_batch_x_y(i, i + batch_size)
			pre = pad_sequences(pre, dtype='float32')
			phr = pad_sequences(phr, dtype='float32')
			post = pad_sequences(post, dtype='float32')
			print(self.model.evaluate(x=[pre, phr, post], y=y, verbose=0))


def main():
	categorisator = PhraseContextCategorisator(100)
	categorisator.train(50)
	categorisator.evaluate(50)


if __name__ == "__main__":
	main()
