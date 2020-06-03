import vectorizators
import numpy as np
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Dense, Flatten
from keras.models import Model


class CategorisatorNN(object):
	def __init__(self):
		self.x_train = None
		self.y_train = None
		self.word2vec = vectorizators.get_pretrained_word2vec()
		self.tokenizer = Tokenizer()
		self.embedding_matrix = None
		self.model = None
		self.MAX_SEQ_LENGTH = 200

	def train(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train

		self.create_embedding_matrix(self.x_train)
		self.model = self.create_model()
		self.model.fit(self.x_train, self.y_train)

	def evaluate(self, x_test, y_test):
		pass

	def create_embedding_matrix(self, x_train):
		self.tokenizer.fit_on_texts(x_train)
		self.x_train = pad_sequences(self.tokenizer.texts_to_sequences(self.x_train), maxlen=self.MAX_SEQ_LENGTH)
		embeddings_index = dict()
		for word in self.word2vec.wv.vocab:
			embeddings_index[word] = self.word2vec.wv.get_vector(word)
		self.embedding_matrix = np.zeros((len(self.tokenizer.word_index)+1, self.word2vec.vector_size))
		for word, i in self.tokenizer.word_index.items():
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				self.embedding_matrix[i] = embedding_vector
		print('Loaded %s word vectors' % len(self.word2vec.wv.vocab))

	def create_model(self):
		sequence_input = Input(shape=(self.MAX_SEQ_LENGTH,))
		embedding_layer = Embedding(len(self.tokenizer.word_index)+1,
									self.word2vec.vector_size,
									weights=[self.embedding_matrix],
									trainable=False)(sequence_input)
		layers = Dense(128, activation="relu")(embedding_layer)
		layers = Flatten()(layers)
		main_output = Dense(1, activation='softplus')(layers)
		model = Model(inputs = sequence_input, outputs = main_output)
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

		print("Finished creating model")
		print(model.summary())
		return model
