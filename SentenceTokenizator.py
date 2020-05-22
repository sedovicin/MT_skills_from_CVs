import nltk


def tokenize_sentence(sentence: str):
	"""Segments each sentence into tokens (words, punctuation etc.).
	Tokens remain in order of appearance in sentence, ie. 'Hello World!' gets
	segmented into ['Hello', 'World', '!']."""
	return nltk.word_tokenize(sentence)


def run(sentences: list):
	"""Takes list of sentences and segments each sentence into tokens (words, punctuation etc.).
	Tokens are not mixed together, but separated for each sentence individually."""
	sentences_tokens = []
	for sentence in sentences:
		sentences_tokens.append(tokenize_sentence(sentence))
	return sentences_tokens
