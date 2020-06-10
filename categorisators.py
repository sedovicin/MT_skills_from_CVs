import numpy as np
from keras.layers import Input, Dense, LSTM
from keras.models import Model

import vectorizators


def y_1d_to_one_hot(y_1d, categories_count):
	"""
	Turns target values (categories) to its one-hot equivalents.

	:param y_1d: y to be transformed
	:type y_1d: list[int]
	:param categories_count: amount of categories available
	:type categories_count: int
	:return: one-hot equivalent of y, a NumPy array
	:rtype: numpy.ndarray
	"""
	one_hot = np.zeros((len(y_1d), categories_count))
	i = 0
	for value in y_1d:
		one_hot[i][value] = 1
		i += 1
	return one_hot


def create_y_from_x(x, dataset):
	"""
	Creates output structure for input.

	:param x: input
	:type x: list[list[str]]
	:param dataset: dictionary that contains target values
	:type dataset: dict[str, int]
	:return: target structure
	:rtype: list[list[int]]
	"""
	print("Creating y...")
	y = list()
	for sentence in x:
		y_sent = list()
		for word in sentence:
			try:
				value = dataset[word]
				y_sent.append(value)
			except KeyError:
				# TODO: if word doesn't exist in dataset, create value based on similar words's values
				y_sent.append(0)
		y.append(y_sent)
	print("Created y.")
	return y


def xy_dict_to_xy_arrays(dictionary):
	"""
	Extracts vectors and values from dictionary.

	:param dictionary: dictionary to be processed
	:type dictionary: dict[str, tuple[numpy.ndarray, int]]
	:return: first is array of vectors, second is list of values
	"""
	x_train_new = list()
	y_train_new = list()
	for tuple_vec_val in dictionary.values():
		x_train_new.append(tuple_vec_val[0])
		y_train_new.append(tuple_vec_val[1])

	return np.array(x_train_new), y_train_new


class CategorisatorNN(object):
	def __init__(self, corpus=None):
		"""
		:param corpus: Optional: path to corpus
		:type corpus: str
		"""
		self.x_train = None
		self.y_train = None
		self.word2vec = vectorizators.get_word2vec(word2vec_path='word2vec.obj', corpus_path=corpus)
		self.model = self.__create_model()

	def train(self, x_train, y_train, batch_size=None, epochs=1):
		"""
		Takes the input and fits it to model.

		:param x_train: list of sentences. Sentence must be given as list of tokens (words).
		:type x_train: list[list[str]]
		:param y_train: same structure as x_train, but instead of word, must contain target data (category)
		:type y_train: list[list[int]]
		:param batch_size: Number of samples per gradient update. If unspecified, batch_size will default to 32.
		:type batch_size: int
		:param epochs: Number of epochs to train the model.
		:type epochs: int
		"""
		self.x_train, self.y_train = xy_dict_to_xy_arrays(self.transform_x_y(x_train, y_train))
		self.y_train = y_1d_to_one_hot(self.y_train, 2)
		print("Fitting model...")
		self.model.fit(
			self.x_train.reshape((self.x_train.shape[0], 1, self.x_train.shape[1])),
			self.y_train,
			batch_size=batch_size,
			epochs=epochs,
			verbose=1)

	def evaluate(self, x_test, y_test):
		"""
		Evaluates the model by predicting output and comparing it with actual one.

		:param x_test: text input set
		:type x_test: list[list[str]]
		:param y_test: actual values of input
		:type y_test: list[list[int]]
		:return: values for loss and accuracy as tuple
		:rtype: tuple[float, float]
		"""
		print("Evaluating test set...")
		x_test_seq, y_test_seq = xy_dict_to_xy_arrays(self.transform_x_y(x_test, y_test))
		y_test_seq = y_1d_to_one_hot(y_test_seq, 2)
		return self.model.evaluate(
			x_test_seq.reshape((x_test_seq.shape[0], 1, x_test_seq.shape[1])),
			y_test_seq, verbose=1)

	def predict(self, x):
		"""
		Predicts output for given input.

		:param x: input to be processed
		:return: output
		"""
		dictionary = self.transform_x(x)
		x_test_words, x_test_vectors = np.array(list(dictionary.keys())), np.array(list(dictionary.values()))
		results = self.model.predict(
			x_test_vectors.reshape((x_test_vectors.shape[0], 1, x_test_vectors.shape[1])),
			verbose=1)
		for i in range(len(x_test_words)):
			print("%s: %s" % (x_test_words[i], results[i]))

	def transform_x(self, x):
		"""
		Transforms x to appropriate shape.
		Replaces every word with its vector, and returns dictionary containing both.

		:param x: x to be processed, list of sentences (list of words)
		:type x: list[list[str]]
		:return: dictionary, where key is the word, and value is vector
		:rtype: dict[str, numpy.ndarray]
		"""
		print("Transforming x...")
		dictionary_vec_val = dict()
		for sentence in x:
			vectors = self.sentence_to_word2vec_vectors(sentence)

			for index in range(len(sentence)):
				dictionary_vec_val[sentence[index]] = vectors[index]

		return dictionary_vec_val

	def transform_x_y(self, x, y):
		"""
		Transforms both x and y to appropriate shape.
		Replaces every word with its vector, and returns dictionary containing both, with value included.
		Makes sure that every word keeps proper value.

		:param x: x to be processed, list of sentences (list of words)
		:type x: list[list[str]]
		:param y: y for given x
		:type y: list[list[int]]
		:return: dictionary, where key is the word, and value is tuple (vector, value)
		:rtype: dict[str, tuple[numpy.ndarray, int]]
		"""

		print("Transforming x and y...")
		dictionary_vec_val = dict()
		i = 0
		for sentence in x:
			vectors = self.sentence_to_word2vec_vectors(sentence)
			values = y[i]
			for index in range(len(sentence)):
				dictionary_vec_val[sentence[index]] = (vectors[index], values[index])
			i += 1
		return dictionary_vec_val

	def sentence_to_word2vec_vectors(self, sentence):
		"""
		Creates list of vectors for given sentence (list of words).
		Vectors for words are fetched from Word2Vec, or set to 0 (all the values) if
		Word2Vec does not contain any of the word in sentence.

		:param sentence: list of words to be processed
		:type sentence: list[str]
		:return: new array containing vectors
		:rtype: numpy.ndarray
		"""
		vectors = np.zeros((len(sentence), self.word2vec.wv.vector_size))
		index = 0
		for word in sentence:
			try:
				vector = self.word2vec.wv.get_vector(word)
			except KeyError:
				vector = self.create_vector_for_oov_word(sentence, index)
			vectors[index] = vector
			index += 1
		return vectors

	def create_vector_for_oov_word(self, words, index):
		"""
		Creates vector for out-of-vocabulary word.
		Uses context (rest of the sentence) of the word to create best matching vector possible.

		:param words: sentence containing context and the word
		:type words: list[str]
		:param index: index of the word to be processed
		:type index: int
		:return: vector for word. If all the context words are out-of-vocabulary for the current model, returns 0-vector.
		:rtype: numpy.ndarray
		"""
		edited = words.copy()
		del edited[index]
		predicted_similar_words = self.word2vec.predict_output_word(edited)

		if predicted_similar_words is None:
			return np.zeros(self.word2vec.wv.vector_size)

		sum_probabs = 0
		for word in predicted_similar_words:
			sum_probabs += word[1]
		factor = 1.0 / sum_probabs
		new_vector = np.zeros(self.word2vec.wv.vector_size)
		for word in predicted_similar_words:
			word_vector = self.word2vec.wv.get_vector(word[0])
			new_vector += (word_vector * word[1] * factor)
		return new_vector

	def __create_model(self):
		"""
		Creates model.

		:return: new model
		:rtype: keras.models.Model
		"""
		print("Creating model...")
		model_input = Input(shape=(None, self.word2vec.vector_size))

		lstm = LSTM(256)(model_input)
		dense = Dense(128, activation='relu')(lstm)
		dense = Dense(64, activation='relu')(dense)
		dense = Dense(16, activation='relu')(dense)
		main_output = Dense(2, activation='softmax')(dense)

		model = Model(inputs=model_input, outputs=main_output)

		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

		print("Finished creating model")
		print(model.summary())
		return model
