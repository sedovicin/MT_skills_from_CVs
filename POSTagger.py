import nltk


def tag_parts_of_speech(sentence_tokens: list):
	"""Analyses words in sentence and tags each word with part-of-speech label"""
	return nltk.pos_tag(sentence_tokens, tagset='universal')


def run(sentences_tokens: list):
	"""Tags each word in each sentence with a tag that represents a part-of-speech label."""
	return nltk.pos_tag_sents(sentences_tokens, tagset='universal')
