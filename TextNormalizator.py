from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


class WordTextNormalizator:
	"""
	Normalizes given words. This can be done using stemming or lemmatization.
	"""
	def __init__(self):
		self.stemmer = PorterStemmer()
		self.lemmatizer = WordNetLemmatizer()

	def stem(self, word):
		"""
		Stems word.

		:param word: word to be stemmed
		:type word: str
		:return: stemmed word
		:rtype: str
		"""
		return self.stemmer.stem(word)

	def lemmatize_word(self, word_pos):
		"""
		Lemmatizes word. Word must be passed as a tuple (word, pos_tag).

		:param word_pos: word and its POS tag to be processed
		:type word_pos: tuple[str, str]
		:return: word in lemmatized form
		:rtype: str
		"""
		try:
			return self.lemmatizer.lemmatize(word_pos[0], self.__transform_pos_for_lemma(word_pos[1]))
		except KeyError:
			return self.lemmatizer.lemmatize(word_pos[0])

	def lemmatize_sentence(self, sentence):
		"""
		Lemmatizes every word in a sentence.

		:param sentence: list of words with its POS tags
		:type sentence: list[tuple[str,str]]
		:return: list of words in lemmatized form
		:rtype: list[str]
		"""
		lemmatized_sentence = list()
		for word in sentence:
			lemmatized_sentence.append(self.lemmatize_word(word))
		return lemmatized_sentence

	@staticmethod
	def __transform_pos_for_lemma(pos_tag):
		"""
		Transforms POS tagger's tags to WordNet tags
		:param pos_tag: POS tag to be transformed
		:type pos_tag: str
		:return: matching WordNet tag
		:rtype: str
		"""
		tags = {
			"ADJ": wordnet.ADJ,
			"ADV": wordnet.ADV,
			"NOUN": wordnet.NOUN,
			"VERB": wordnet.VERB
		}
		return tags.get(pos_tag)
