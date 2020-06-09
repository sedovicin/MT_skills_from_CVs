import vectorizators
import numpy as np
from keras_preprocessing.text import Tokenizer
from keras.layers import Input, Dense, LSTM
from keras.models import Model


def y_to_one_hot(y, categories_count):
	"""
	Turns target value (categories) to its one-hot equivalent.

	:param y: y to be transformed
	:type y: list[list[int]]
	:param categories_count: amount of categories available
	:type categories_count: int
	:return: one-hot equivalent of y, a NumPy array
	:rtype: numpy.ndarray
	"""
	one_hot = np.zeros(len(y))
	i = 0
	for sentence in y:
		one_hot_sent = np.zeros((len(sentence), categories_count))
		j = 0
		for value in sentence:
			one_hot_sent[j][value] = 1
			j += 1
		one_hot[i] = one_hot_sent
		i += 1
	return one_hot


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
		print("Creating x_train...")
		self.x_train = self.sentences_to_word2vec_vectors(x_train)
		print("Creating y_train...")
		self.y_train = y_to_one_hot(y_train, 2)
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
		x_test_seq = self.sentences_to_word2vec_vectors(x_test)
		y_test_seq = y_to_one_hot(y_test, 2)
		return self.model.evaluate(
			x_test_seq.reshape((x_test_seq.shape[0], 1, x_test_seq.shape[1])),
			y_test_seq, verbose=1)

	def predict(self, x):
		"""
		Predicts output for given input

		:param x: input to be processed
		:return: output
		"""
		x_test_seq = self.words_to_word2vec_vectors(x)
		results = self.model.predict(
			x_test_seq.reshape((x_test_seq.shape[0], 1, x_test_seq.shape[1])),
			verbose=1)
		print(results)

	def sentences_to_word2vec_vectors(self, sentences):
		"""
		Creates list of vectors for all the words from given sentences.
		Vectors for words are fetched from Word2Vec, or set to 0 (all the values) if
		Word2Vec does not contain the processed word.

		:param sentences: list of sentences to be processed
		:type sentences: list[list[str]]
		:return: arrays containing all vectors
		:rtype: numpy.ndarray
		"""
		print("Transforming all words to vectors...")
		vectors_all = np.array((None, self.word2vec.wv.vector_size))
		for sentence in sentences:
			vectors = self.words_to_word2vec_vectors(sentence)
			np.concatenate((vectors_all, vectors), out=vectors_all)
		print("Finished transforming words to vectors.")
		return vectors_all

	def words_to_word2vec_vectors(self, words):
		"""
		Creates list of vectors for given words.
		Vectors for words are fetched from Word2Vec, or set to 0 (all the values) if
		Word2Vec does not contain the processed word.

		:param words: list of words to be processed
		:type words: list[str]
		:return: new array containing vectors
		:rtype: numpy.ndarray
		"""
		vectors = np.zeros((len(words), self.word2vec.wv.vector_size))
		index = 0
		for word in words:
			try:
				vector = self.word2vec.wv.get_vector(word)
			except KeyError:
				vector = self.create_vector_for_oov_word(words, index)
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
