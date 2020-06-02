import vectorizators
import numpy as np
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Dense
from keras.models import Model


class CategorisatorNN(object):
	def __init__(self):
		self.x_train = None
		self.y_train = None
		self.word2vec = vectorizators.get_pretrained_word2vec()
		self.tokenizer = Tokenizer()
		self.embedding_matrix = None
		self.model = None

	def train(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train

		self.get_embedding_matrix(x_train)
		self.model = self.create_model(x_train)

		self.model.fit(x_train, y_train)

	def create_embedding_matrix(self, x_train):
		self.tokenizer.fit_on_texts(x_train)
		x_train_seq = pad_sequences(self.tokenizer.texts_to_sequences(self.x_train))
		list_tokenizer_train = self.tokenizer.texts_to_sequences()
		embeddings_index = dict()
		for word in self.word2vec.wv.vocab:
			embeddings_index[word] = self.word2vec.wv.get_vector(word)
		self.embedding_matrix = np.zeros((len(self.tokenizer.word_index), self.word2vec.vector_size))
		for word, i in self.tokenizer.word_index.items():
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				self.embedding_matrix[i] = embedding_vector
		print('Loaded %s word vectors' % len(self.word2vec.wv.vocab))

	def create_model(self, x_train):
		sequence_input = Input()
		embedding_layer = Embedding(len(self.tokenizer.word_index),
									self.word2vec.vector_size,
									weights=[self.embedding_matrix],
									trainable=False)(sequence_input)
		layers = Dense(128, activation="relu")(embedding_layer)
		main_output = Dense(2, activation='softplus')(layers)
		model = Model(inputs = sequence_input, outputs = main_output)
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

		print("Finished creating model")
		print(model.summary())
		return model
