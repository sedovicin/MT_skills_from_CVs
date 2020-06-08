from unittest import TestCase
import SentenceTokenizator as st


class Test(TestCase):
	def test_tokenize_sentence(self):
		self.assertEqual(TEST_RESULT_SENT, st.tokenize_sentence(TEST_SENTENCE))

	def test_multiple_sentences(self):
		self.assertEqual(TEST_RESULT_TWO_SENTS, st.tokenize_sentences(TEST_TWO_SENTS))


TEST_SENTENCE = "This is a test sentence, with some sp€cial chars C++ and punctuation."
TEST_RESULT_SENT = ["This", "is", "a", "test", "sentence", ",", "with", "some", "sp€cial", "chars", "C++",
					"and", "punctuation", "."]
TEST_TWO_SENTS = [TEST_SENTENCE, "Another sentence here."]
TEST_RESULT_SECOND_SENT = ["Another", "sentence", "here", "."]
TEST_RESULT_TWO_SENTS = [TEST_RESULT_SENT, TEST_RESULT_SECOND_SENT]
