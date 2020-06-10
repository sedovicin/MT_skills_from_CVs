import json

import POSTagger
import SentenceTokenizator as Tokenizator
import TextFromFileExtractor as Extractor
import TextSegmentator as Segmentator
import vectorizators
from TextNormalizator import WordTextNormalizator

NOT_SKILL = 0
SKILL = 1


def remove_punctuation(tagged_sentences):
	"""
	Removes punctuation from list of words with their tags as they are not really necessary in further steps.

	:param tagged_sentences: sentences to be processed
	:type tagged_sentences: list[list[tuple]]
	:return: new list without punctuation
	"""
	clean = list()
	for tagged_sentence in tagged_sentences:
		clean_tagged_sentence = list()
		for word_tag in tagged_sentence:
			if word_tag[1] != '.':
				clean_tagged_sentence.append(word_tag)
		clean.append(clean_tagged_sentence)
	return clean


def file_to_tokens(file):
	"""
	Extracts all sentences from file as list of tokens (words).

	:param file: path of file to be processed
	:type file: str
	:return: List of sentences, and sentence is a list of tokens
	:rtype: list[list[str]]
	"""
	text = Extractor.extract(file)
	cleaned_text = Extractor.remove_unsupported_chars(text)
	sentences = Segmentator.segment(cleaned_text)
	return Tokenizator.tokenize_sentences(sentences)


def add_to_dataset(tokens, dictionary, value, overwrite=True, lemmatize=False):
	"""
	Adds every word in given list of list of tokens to given dictionary with given value.

	:param tokens: path to file to be processed
	:type tokens: list[list[str]]
	:param dictionary: dictionary to be filled
	:type dictionary: dict
	:param value: value to be set for every word in file
	:type value: int
	:param overwrite: If true, overwrites existing value with new one.
	:type overwrite: bool
	:param lemmatize: If true, lemmatizes word before putting it to dictionary
	:type lemmatize: bool
	"""
	words = list()
	if lemmatize:
		pos_tags_sentences = POSTagger.tag_pos_sentences(tokens)
		pos_tags_sents_clean = remove_punctuation(pos_tags_sentences)
		normalizator = WordTextNormalizator()
		lemmatized_words = list()
		for sentence in pos_tags_sents_clean:
			lemmatized_words.extend(normalizator.lemmatize_sentence(sentence))
		words = lemmatized_words
	else:
		for sentence_tokens in tokens:
			words.extend(sentence_tokens)
	for word in words:
		if overwrite or (word not in dictionary):
			dictionary[word] = value


def import_dataset(file):
	"""
	Imports dataset from file. File must be readable by json.

	:param file: path of file to be processed
	:type file: str
	:return: Object the file is parsed into
	"""
	print("Importing dataset from %s..." % file)
	with open(file, 'r', encoding='utf8') as fp:
		json_file = json.load(fp)
	return json_file


def export_as_json(data, file):
	"""
	Exports given data to given file as json.

	:param data: data to be exported
	:type data: Any
	:param file: path of file into which the data is exported
	:type file: str
	"""
	print("Exporting data to %s..." % file)
	with open(file, 'w', encoding='utf8') as fp:
		json.dump(data, fp)
	print("Finished exporting data.")


def create_dataset_corpus(prefix, begin, end):
	dataset = dict()
	corpus = vectorizators.get_gutenberg_corpus()
	for i in range(begin, end):
		if i % 100 == 0:
			print("Processing file no. %s..." % str(i))

		tokens = file_to_tokens('cv_extracted/cvs/%s_cv.txt' % str(i + 1))
		add_to_dataset(tokens, dataset, NOT_SKILL, overwrite=False)
		corpus.extend(tokens)

		tokens_skill = file_to_tokens('cv_extracted/skills/%s_skills.txt' % str(i + 1))
		add_to_dataset(tokens_skill, dataset, SKILL)
	export_as_json(dataset, 'dataset_%s.json' % prefix)
	export_as_json(corpus, 'corpus_%s.json' % prefix)


def main():
	pass


if __name__ == "__main__":
	main()
