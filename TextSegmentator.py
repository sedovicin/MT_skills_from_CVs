import nltk


def segment_by_lines(text: str):
	"""
	Segments text into sentences by lines, each line being one sentence.

	:param text: text to be processed
	:type text: str
	:return: list of lines
	:rtype: list[str]
	"""
	return text.splitlines()


def segment_by_punctuation(text: str):
	"""
	Segments text into sentences by punctuation.

	:param text: text to be processed
	:type text: str
	:return: sentences
	"""
	return nltk.sent_tokenize(text)


def segment(text: str):
	"""
	Segments text to sentences, first by lines, followed by punctuation.

	:param text: text to be processed
	:type text: str
	:return: list of sentences
	:rtype: list[str]
	"""
	lines = segment_by_lines(text)
	sentences = []
	for line in lines:
		stripped_line = line.strip()
		if stripped_line != '':
			sentences.extend(segment_by_punctuation(stripped_line))
	return sentences
