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
	def __init__(self, corpus_path, dataset_path):
		self.vectorizator = vectorizators.Word2VecVectorizator()
		self.parser = PhraseParser()

	def get_batch_x_y(self, begin, end):
		"""
		Creates samples and its targets from files indexed begin index (incl.) to end index (excl.)
		:param begin:
		:param end:
		:return:
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

	@staticmethod
	def phrases_context_as_vectors(sentence, vectors, categories):
		index = 0  # pointer to word that is being currently processed throughout the code
		begin = 0  # pointer to beginning of matching phrase
		end = 1  # pointer to word following the end of matching phrase,
		pre_contexts = []
		phrases = []
		post_contexts = []
		ys = []
		sent_with_parsed_tags = sentence.pos()

		while begin < len(sent_with_parsed_tags):
			# FIND NEXT PHRASE INDEX
			index = begin
			while (index < len(sent_with_parsed_tags)) and (sent_with_parsed_tags[index][1] != 'PHRASE'):
				index += 1
			if index >= len(sent_with_parsed_tags):
				break
			begin = index

			# GET CATEGORY OF FIRST WORD IN PHRASE
			category = categories.get(sent_with_parsed_tags[begin][0][0], 0)

			# FIND END OF PHRASE
			while True:
				index += 1
				if (index >= len(sent_with_parsed_tags)) \
					or (sent_with_parsed_tags[index][1] != 'PHRASE') \
					or (category != categories.get(sent_with_parsed_tags[begin][0][0], 0)):
					break
			end = index

			# GET PRE-PHASE CONTEXT
			pre_phrase_context = []
			index = begin - 1
			while (index >= 0) and ((begin - index) <= 3):
				pre_phrase_context.append(vectors[index])
				index -= 1
			pre_phrase_context.reverse()

			# GET POST-PHASE CONTEXT
			post_phrase_context = []
			index = end
			while (index < len(sent_with_parsed_tags)) and ((index - end) < 3):
				post_phrase_context.append(vectors[index])
				index += 1

			# GET PHRASE
			phrase = []
			index = begin
			while index < end:
				phrase.append(vectors[index])
				index += 1

			pre_contexts.append(np.array(pre_phrase_context))
			phrases.append(np.array(phrase))
			post_contexts.append(np.array(post_phrase_context))

			y_entry = np.zeros(2)
			y_entry[category] = 1
			ys.append(y_entry)

			# Moving on...
			begin = end
		return pre_contexts, phrases, post_contexts, ys


def main():
	pass


if __name__ == "__main__":
	main()
