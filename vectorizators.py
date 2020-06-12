import nltk.corpus
from sklearn.feature_extraction.text import TfidfVectorizer
from pickle import Pickler
import pickle as pic
from nltk.text import TextCollection
import numpy as np


class TFIDFVectorizator:

	COLLECTION_FILE_NAME = "text_collection.obj"

	def __init__(self):
		corpus = [nltk.corpus.gutenberg.raw(f) for f in
				  nltk.corpus.gutenberg.fileids()]  # TODO: add a set of cvs here as well
		#corpus = ["Hello. What's up? I'm fine.", "I'm in love with a beautiful girl.",
		#		  "What is your problem, my love?"]
		try:
			with open(self.COLLECTION_FILE_NAME, 'rb') as collection_file:
				self.vectorizer = pic.load(collection_file)

				print("Loaded collection from file.")
		except FileNotFoundError:
			print("Collection file not found, creating new...")
			self.vectorizer = TfidfVectorizer(stop_words=None)
			self.vectorizer.fit(corpus)
			with open(self.COLLECTION_FILE_NAME, 'wb') as collection_file:
				pic.dump(self.vectorizer, collection_file)
				print("Collection file created and collection loaded")
		self.matrix = self.vectorizer.transform(corpus).todense()
		self.feature_names = self.vectorizer.get_feature_names()

		self.word_tfidf = dict()
		for word in self.vectorizer.vocabulary_.items(): # FIXME: is the score inverse maybe?
			docs = np.squeeze(np.asarray(self.matrix[:, int(word[1])]))
			scores = [score for score in docs if not score < 0.0000000001]  # remove scores if 0 or very close to 0
			self.word_tfidf[word[0]] = sum(scores) / len(scores)  # average TF-IDF score

	def vectorize(self, word):
		"""Returns TF-IDF score for given word, or 0 if the word does not exist in vocabulary."""
		return self.word_tfidf.get(word, 0.0)


class Word2VecVectorizator:
	def __init__(self):
		self.word2vec = self.__get_word2vec(word2vec_path='word2vec.obj')
		self.additional_vector_set = dict()

	@staticmethod
	def __get_word2vec(word2vec_path=None, corpus_path=None):
		"""
		Loads pickled Word2Vec model from path. If the file doesn't exist, creates a new trained one using given corpus.
		If the corpus is None, uses default corpus (Gutenberg).

		:param word2vec_path: path to file containing Word2Vec model
		:type word2vec_path: str
		:param corpus_path: path to file containing corpus
		:type corpus_path: str
		:return: Word2Vec trained object
		:rtype: gensim.models.Word2Vec
		"""
		from gensim.models import Word2Vec
		if word2vec_path is not None:
			try:
				with open(word2vec_path, 'rb') as collection_file:
					word2vec = pic.load(collection_file)
					print("Loaded Word2Vec from file.")
			except FileNotFoundError:
				print("Word2Vec file not found, creating new...")
				if corpus_path is None:
					corpus = get_gutenberg_corpus()
					print("Got default corpus.")
				else:
					import json
					try:
						with open(corpus_path, 'r', encoding='utf8') as fp_corpus:
							corpus = json.load(fp_corpus)
							print("Loaded corpus from file.")
					except FileNotFoundError:
						corpus = get_gutenberg_corpus()
						print("Corpus file not found, loading default corpus.")
				word2vec = Word2Vec(min_count=0, iter=10)
				word2vec.build_vocab(sentences=corpus)
				word2vec.train(sentences=corpus, total_examples=word2vec.corpus_count, epochs=10)
				with open(word2vec_path, 'wb') as collection_file:
					pic.dump(word2vec, collection_file)
					print("Word2Vec file created and loaded")
		else:
			raise ValueError("Path to file must be given so the method knows where to find or to save model.")
		return word2vec

	def get_vectors(self, sentence):
		"""
		Gets vector for each word in sentence. If vector for word does not exist,
		creates one based on the context of sentence.
		:param sentence:
		:return: array of vectors
		"""
		vectors = np.zeros((len(sentence), self.word2vec.wv.vector_size))
		index = 0
		for word in sentence:
			try:
				vector = self.word2vec.wv.get_vector(word)
			except KeyError:
				try:
					vector = self.additional_vector_set[word]
				except KeyError:
					vector = self.create_vector_for_oov_word(sentence, index=index)
					self.additional_vector_set[word] = vector
			vectors[index] = vector
			index += 1
		return vectors

	def create_vector_for_oov_word(self, sentence, index):
		"""
		Creates vector for out-of-vocabulary word.
		Uses context (rest of the sentence) of the word to create best matching vector possible.

		:param sentence: sentence containing context and the word
		:type sentence: list[str]
		:param index: index of the word to be processed
		:type index: int
		:return: vector for word. If all the context words are out-of-vocabulary for the current model, returns 0-vector.
		:rtype: numpy.ndarray
		"""
		edited = sentence.copy()
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


def get_gutenberg_corpus():
	"""
	:returns: corpus made from NLTK Gutenberg files, in shape of sentences.
	:rtype: list
	"""
	import nltk
	corpus = []
	for f in nltk.corpus.gutenberg.fileids():
		corpus.extend(nltk.corpus.gutenberg.sents(f))
	return corpus
