import nltk
import numpy as np
from nltk.tree import Tree

import POSTagger
import dataset_corpus_mgmt as dc_mgmt
import vectorizators


class PhraseParser:
	"""
	Used for extracting only requested parts of sentence, considering Part-Of-Speech tags.
	"""
	def __init__(self, grammar=None):
		self.grammar = grammar
		if self.grammar is None:
			self.grammar = """
				PHRASE: {<VERB>?<ADJ>*<NOUN>+<VERB>?}
					{<VERB>?<ADJ>+<CONJ>+<ADJ>*<NOUN>+<VERB>?}
				"""
		self.parser = nltk.RegexpParser(self.grammar)

	def parse(self, tagged_sentence):
		"""
		Parses sentence.

		:param tagged_sentence: sentence to be parsed
		:return: Parsed sentence as tree
		:rtype: Tree
		"""
		return self.parser.parse(tagged_sentence)

	def parse_sents(self, tagged_sentences):
		"""
		Parses multiple sentences

		:param tagged_sentences: sentences to be parsed
		:return: Parsed sentence as list of trees
		:rtype: list[Tree]
		"""
		sentences_parsed = []
		for sentence in tagged_sentences:
			sentences_parsed.append(self.parse(sentence))
		return sentences_parsed


class SampleGenerator:
	"""
	Extracts samples from CVs and skills. Uses files.
	"""
	def __init__(self, context_size, categories_count):
		self.vectorizator = vectorizators.Word2VecVectorizator()
		self.parser = PhraseParser()
		self.context_size = context_size
		self.categories_count = categories_count

	def get_batch_x_y(self, begin, end):
		"""
		Creates single batch of inputs and targets using chunk of files.

		:param begin: index of first file to be extracted
		:param end: last extracting file is this parameter minus one, so end-1
		:return: batch of input and output, arrays: pre-phrase context, phrase, post-phrase context, target (categories)
		"""
		pre_contexts = []
		phrases = []
		post_contexts = []
		ys = []
		for i in range(begin, end):
			cv_sentences = dc_mgmt.file_to_tokens('cv_extracted/cvs/%s_cv.txt' % i)

			categories = {}
			sentences_skills = dc_mgmt.file_to_tokens('cv_extracted/skills/%s_skills.txt' % i)
			for sentence in sentences_skills:
				for word in sentence:
					categories[word] = 1

			for sentence in cv_sentences:
				tag_sentence = POSTagger.tag_pos(sentence)
				parsed_sentence = self.parser.parse(tag_sentence)

				vectors = self.vectorizator.get_vectors(sentence)

				pre_context, phrase, post_context, y = \
					self.phrases_context_as_vectors(parsed_sentence, vectors, categories)
				pre_contexts.extend(pre_context)
				phrases.extend(phrase)
				post_contexts.extend(post_context)
				ys.extend(y)
		return np.array(pre_contexts), np.array(phrases), np.array(post_contexts), np.array(ys)

	def phrases_context_as_vectors(self, sentence, vectors, categories):
		"""
		Extracts phrases and context from given sentence.
		:param sentence: sentence to be extracted from
		:type sentence: Tree
		:param vectors: array of vectors for given sentence (one word = one vector)
		:type vectors: numpy.ndarray
		:param categories: dictionary of words (key) and its category (value)
		:return: batch of input and output for given sentence,
			arrays: pre-phrase context, phrase, post-phrase context, target (categories)
		"""
		index = 0  # pointer to word that is being currently processed throughout the code
		begin = 0  # pointer to beginning of matching phrase
		end = 1  # pointer to word following the end of matching phrase,
		pre_contexts = []
		phrases = []
		post_contexts = []
		ys = []
		sent_with_parsed_tags = sentence.pos()
		word_count = len(sent_with_parsed_tags)

		while begin < word_count:
			# FIND NEXT PHRASE INDEX
			index = begin
			while (index < word_count) and (sent_with_parsed_tags[index][1] != 'PHRASE'):
				index += 1
			if index >= word_count:
				break
			begin = index

			# GET CATEGORY OF FIRST WORD IN PHRASE
			category = categories.get(sent_with_parsed_tags[begin][0][0], 0)

			# FIND END OF PHRASE
			while True:
				index += 1
				if (index >= word_count) \
					or (sent_with_parsed_tags[index][1] != 'PHRASE') \
					or (category != categories.get(sent_with_parsed_tags[begin][0][0], 0)):
					break
			end = index

			# GET PRE-PHASE CONTEXT
			index = begin - self.context_size
			pre_phrase_context = self.vectorizator.get_empty_vector_array(self.context_size)
			while index < begin:
				if index >= 0:
					pre_phrase_context[index - begin + self.context_size] = vectors[index]
				index += 1

			# GET PHRASE
			phrase = self.vectorizator.get_empty_vector_array(end - begin)
			while index < end:
				phrase[index - begin] = vectors[index]
				index += 1

			# GET POST-PHASE CONTEXT
			post_phrase_context = self.vectorizator.get_empty_vector_array(self.context_size)
			while index < (end + self.context_size):
				if index < word_count:
					pre_phrase_context[index - end] = vectors[index]
				index += 1

			pre_contexts.append(pre_phrase_context)
			phrases.append(phrase)
			post_contexts.append(post_phrase_context)

			y_entry = np.zeros(self.categories_count)
			y_entry[category] = 1
			ys.append(y_entry)

			# Moving on...
			begin = end
		return pre_contexts, phrases, post_contexts, ys


def main():
	pass


if __name__ == "__main__":
	main()
