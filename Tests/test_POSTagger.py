from unittest import TestCase
import POSTagger as post


class Test(TestCase):
	def test_tag_parts_of_speech(self):
		self.assertEqual(SENTENCE_ONE_RES, post.tag_pos(SENTENCE_ONE))

	def test_run(self):
		self.assertEqual([SENTENCE_ONE_RES, SENTENCE_TWO_RES], post.tag_pos_sentences([SENTENCE_ONE, SENTENCE_TWO]))


SENTENCE_ONE = ["This", "is", "a", "test", "sentence", "."]
SENTENCE_ONE_RES = [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('test', 'NN'), ('sentence', 'NN'), ('.', '.')]
SENTENCE_TWO = ["business", "proposition"]
SENTENCE_TWO_RES = [('business', 'NN'), ('proposition', 'NN')]
