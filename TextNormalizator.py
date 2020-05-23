from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

stemmer = PorterStemmer();


def stem(word: str):
	return stemmer.stem(word)


lemmatizer = WordNetLemmatizer()


def lemmatize(word_pos: tuple):
	try:
		result = lemmatizer.lemmatize(word_pos[0], transform_pos_for_lemma(word_pos[1]))
	except KeyError:
		result = lemmatizer.lemmatize(word_pos[0])
	return result


def transform_pos_for_lemma(pos_tag: str):
	tags = {
		"ADJ": wordnet.ADJ,
		"ADV": wordnet.ADV,
		"NOUN": wordnet.NOUN,
		"VERB": wordnet.VERB
	}
	return tags.get(pos_tag, wordnet.NOUN)
