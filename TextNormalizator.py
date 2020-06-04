from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


class WordTextNormalizator:
	"""Normalizes given words. This can be done using stemming or lemmatization."""
	def __init__(self):
		self.stemmer = PorterStemmer()
		self.lemmatizer = WordNetLemmatizer()

	def stem(self, word):
		"""Stems word.

		:type word: str"""
		return self.stemmer.stem(word)

	def lemmatize_word(self, word_pos):
		"""Lemmatizes word. Word must be passed as a tuple, where left value is the word itself,
		and second value is word's POS tag.

		:type word_pos: tuple"""
		try:
			return self.lemmatizer.lemmatize(word_pos[0], self.__transform_pos_for_lemma(word_pos[1]))
		except KeyError:
			return self.lemmatizer.lemmatize(word_pos[0])

	def lemmatize_sentence(self, sentence):
		"""Lemmatizes every word in a sentence. See lemmatize_word.

		:type sentence: list[tuple]
		"""
		lemmatized_sentence = list()
		for word in sentence:
			lemmatized_sentence.append(self.lemmatize_word(word))
		return lemmatized_sentence

	@staticmethod
	def __transform_pos_for_lemma(pos_tag: str):
		tags = {
			"ADJ": wordnet.ADJ,
			"ADV": wordnet.ADV,
			"NOUN": wordnet.NOUN,
			"VERB": wordnet.VERB
		}
		return tags.get(pos_tag)
