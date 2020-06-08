import nltk


def tokenize_sentence(sentence):
	"""
	Segments each sentence into tokens (words, punctuation etc.).
	Tokens remain in order of appearance in sentence, for example: 'Hello World!' gets
	segmented into ['Hello', 'World', '!'].

	:param sentence: sentence to be processed
	:type sentence: str
	:return: sentence as list of words
	:rtype: list[str]
	"""
	return nltk.word_tokenize(sentence)


def tokenize_sentences(sentences):
	"""
	Takes list of sentences and segments each sentence into tokens (words, punctuation etc.).
	Tokens are not mixed together, but separated for each sentence individually. In other words,
	every sentence becomes a list of words.

	:param sentences: list of sentences to be processed
	:type sentences: list[str]
	:return: list of tokenized sentences
	:rtype: list[list[str]]
	"""
	sentences_tokens = []
	for sentence in sentences:
		sentences_tokens.append(tokenize_sentence(sentence))
	return sentences_tokens
