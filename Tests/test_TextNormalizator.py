from unittest import TestCase
from TextNormalizator import WordTextNormalizator

class Test(TestCase):

	def test_stem(self):
		self.assertEqual('skill', WordTextNormalizator().stem('skills'))

	def test_lemmatize_word(self):
		self.assertEqual('provide', WordTextNormalizator().lemmatize_word(WORD1))

	def test_lemmatize_sentence(self):
		self.assertEqual(SENTENCE_LEMMATIZED, WordTextNormalizator().lemmatize_sentence(SENTENCE))


SENTENCE = [('organization', 'NOUN'), ('provides', 'VERB'), ('me', 'PRON'), ('the', 'DET'), ('opportunity', 'NOUN'),
			('to', 'PRT'), ('improve', 'VERB'), ('my', 'PRON'), ('skills', 'NOUN')]
SENTENCE_LEMMATIZED = ['organization', 'provide', 'me', 'the', 'opportunity', 'to', 'improve', 'my', 'skill']
WORD1 = ('provides', 'VERB')
WORD2 = ('skills', 'NOUN')
