import nltk


def tag_pos(sentence_tokens):
	"""
	Analyses words in sentence and tags each word with part-of-speech label. Uses universal tagset.

	:param sentence_tokens: list of tokens (words) to be processed
	:type sentence_tokens: list[str]
	:return: list of tuples (word, pos_tag)
	:rtype: list[tuple[str,str]]
	"""
	return nltk.pos_tag(sentence_tokens, tagset='universal')


def tag_pos_sentences(sentences_tokens):
	"""
	Tags each word in each sentence with a tag that represents a part-of-speech label.

	:param sentences_tokens: list of sentences (list of tokens (words)) to be processed
	:type sentences_tokens: list[list[str]]
	:return: list of list of tuples (word, pos_tag)
	:rtype: list[list[tuple[str,str]]]
	"""
	return nltk.pos_tag_sents(sentences_tokens, tagset='universal')
