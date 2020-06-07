import vectorizators
import numpy as np
from keras_preprocessing.text import Tokenizer
from keras.layers import Input, Dense, LSTM
from keras.models import Model


def y_to_one_hot(y, categories_count):
	"""
	Turns regular y (categories) to its one-hot equivalent.

	:param y: y to be transformed
	:type y:
	:type categories_count: int
	:param categories_count: amount of categories available
	:return: one-hot equivalent of y, a 2D NumPy array
	:rtype: numpy.ndarray
	"""
	one_hot = np.zeros((len(y), categories_count))
	i = 0
	for value in y:
		one_hot[i][value] = 1
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
		self.embedding_matrix = None
		self.model = None
		self.tokenizer = Tokenizer()

	def train(self, x_train, y_train):
		# self.x_train = x_train
		self.y_train = y_train

		# self.create_embedding_matrix(x_train)
		self.model = self.create_model()
		print("Creating x_train...")
		self.x_train = self.words_to_word2vec_vectors(x_train)
		print("Creating y_train...")
		self.y_train = y_to_one_hot(self.y_train, 2)
		print("Fitting model...")
		self.model.fit(
			self.x_train.reshape((self.x_train.shape[0], 1, self.x_train.shape[1])),
			self.y_train,
			batch_size=1000,
			epochs=3,
			verbose=1)

	def evaluate(self, x_test, y_test):
		"""
		Must give the sentences!
		:param x_test:
		:param y_test:
		:return:
		"""
		x_test_seq = self.words_to_word2vec_vectors(x_test)
		y_test_seq = y_to_one_hot(y_test, 2)
		results = self.model.evaluate(
			x_test_seq.reshape((x_test_seq.shape[0], 1, x_test_seq.shape[1])),
			y_test_seq, verbose=1)
		print("loss, acc", results)

	def predict(self, x):
		"""
		Must give the sentences!
		:param x:
		:return:
		"""
		x_test_seq = self.words_to_word2vec_vectors(x)
		results = self.model.predict(
			x_test_seq.reshape((x_test_seq.shape[0], 1, x_test_seq.shape[1])),
			verbose=1)
		print(results)

	def sentences_to_word2vec_vectors(self, sentences):
		vectors_all = np.array((None, 100))
		for sentence in sentences:
			vectors = self.words_to_word2vec_vectors(sentence)
			np.concatenate((vectors_all, vectors), out=vectors_all)

	def words_to_word2vec_vectors(self, words):
		"""Creates list of vectors for given words.
		Vectors for words are fetched from Word2Vec, or set to 0 (all the values) if
		Word2Vec does not contain the processed word.

		:param words: list of words to be processed
		:type words: list
		:return: new array containing vectors
		:rtype: numpy.ndarray
		"""
		print("Turning words to vectors...")
		vectors = np.zeros((len(words), self.word2vec.wv.vector_size))
		index = 0
		for word in words:
			try:
				vector = self.word2vec.wv.get_vector(word)
			except KeyError:
				vector = self.create_vector_for_oov_word(words, index)
			vectors[index] = vector
			index += 1
		print("Finished turning words to vectors.")
		return vectors

	def create_vector_for_oov_word(self, words, index):
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

	def create_model(self):
		print("Creating model...")
		model_input = Input(shape=(None, self.word2vec.vector_size))

		lstm = LSTM(256)(model_input)
		dense = Dense(128, activation='relu')(lstm)
		dense = Dense(64, activation='relu')(dense)
		dense = Dense(16, activation='relu')(dense)
		main_output = Dense(2, activation='softplus')(dense)

		model = Model(inputs=model_input, outputs=main_output)

		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

		print("Finished creating model")
		print(model.summary())
		return model
