import vectorizators
import numpy as np
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Dense, Flatten, LSTM, concatenate
from keras.models import Model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class CategorisatorNN(object):
	def __init__(self, corpus=None):
		"""
		:param corpus: Optional: path to corpus
		:type corpus: str
		"""
		self.x_train = None
		self.y_train = None
		self.word2vec = vectorizators.get_word2vec(word2vec_path='word2vec.obj', corpus_path=corpus)
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
		# TODO: add evaluation part
		x_seq = pad_sequences(self.tokenizer.texts_to_sequences(x_test), maxlen=self.MAX_SEQ_LENGTH)

		confidences = self.model.predict(x_seq, verbose=1)

	def predict(self, x):
		x_seq = pad_sequences(self.tokenizer.texts_to_sequences(x), maxlen=self.MAX_SEQ_LENGTH)

		confidences = self.model.predict(x_seq, verbose=1)

	def create_embedding_matrix(self, x_train):
		"""
		Creates embedding matrix. Fills it with vectors from word2vec for each word in training set, if the word exists.
		If not, skips the word.
		:param x_train: input sentences for training
		:type x_train:
		"""
		self.tokenizer.fit_on_texts(x_train)
		self.x_train = pad_sequences(self.tokenizer.texts_to_sequences(self.x_train), maxlen=self.word2vec.vector_size)
		embeddings_index = dict()
		for word in self.word2vec.wv.vocab:
			embeddings_index[word] = self.word2vec.wv.get_vector(word)
		# added 1 because index 0 is reserved for masking
		self.embedding_matrix = np.zeros((len(self.tokenizer.word_index) + 1, self.word2vec.vector_size))
		count_words = 0
		for word, i in self.tokenizer.word_index.items():
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				self.embedding_matrix[i] = embedding_vector
				count_words = count_words + 1
		print('Loaded %s word vectors' % count_words)

	def create_model(self):
		# input_phrase = Input(shape=(None, self.word2vec.vector_size))
		# input_context = Input(shape=(None, self.word2vec.vector_size))
		# input_ph_cont = Input(shape=(2,))

		emb_phrase = Embedding(len(self.tokenizer.word_index),
									self.word2vec.vector_size,
									weights=[self.embedding_matrix],
									trainable=False)(input_phrase)
		emb_context = Embedding(len(self.tokenizer.word_index) + 1,
									self.word2vec.vector_size,
									weights=[self.embedding_matrix],
									trainable=False)(input_context)
		emb_ph_cont = Embedding(len(self.tokenizer.word_index) + 1,
									self.word2vec.vector_size,
									weights=[self.embedding_matrix],
									trainable=False)(input_ph_cont)

		lstm_phrase = LSTM(512)(emb_phrase)
		lstm_context = LSTM(256)(emb_context)
		dense_ph_cont = Dense(512, activation='relu')(emb_ph_cont)

		dense_ph_2 = Dense(128, activation='relu')(lstm_phrase)
		dense_cont_2 = Dense(128, activation='relu')(lstm_context)
		dense_ph_cont_2 = Dense(256, activation='relu')(dense_ph_cont)

		x = concatenate([dense_ph_2, dense_cont_2, dense_ph_cont_2])

		x = Dense(128, activation='relu')(x)
		x = Dense(64, activation='relu')(x)
		x = Dense(32, activation='relu')(x)

		main_output = Dense(2, activation='softplus')(x)

		model = Model(inputs=[input_phrase, input_context, input_ph_cont], outputs=main_output)

		#sequence_input = Input(shape=(self.MAX_SEQ_LENGTH,))
		# embedding_layer = Embedding(len(self.tokenizer.word_index)+1,
		# 							self.word2vec.vector_size,
		# 							weights=[self.embedding_matrix],
		# 							trainable=False)(sequence_input)
		# layers = Dense(128, activation="relu")(embedding_layer)
		# layers = Flatten()(layers)
		# main_output = Dense(1, activation='softplus')(layers)
		# model = Model(inputs=sequence_input, outputs=main_output)

		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

		print("Finished creating model")
		print(model.summary())
		return model
