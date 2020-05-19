def segment_by_lines(text: str):
	"""Segments text into sentences by lines,
	each line being one sentence."""
	return text.splitlines()


def segment_by_punctuation(text: str):
	"""Segments text into sentences by punctuation."""
	# TODO: implement this
	segments = [text]
	return segments


def run(text: str):
	"""Segments text to sentences."""
	lines = segment_by_lines(text)
	sentences = []
	for line in lines:
		stripped_line = line.strip();
		if stripped_line != '':
			sentences.append(segment_by_punctuation(stripped_line))
	return sentences
