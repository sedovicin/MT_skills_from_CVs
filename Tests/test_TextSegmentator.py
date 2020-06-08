from unittest import TestCase
import TextSegmentator as ts


class Test(TestCase):
	def test_segment_by_lines(self):
		self.assertEqual(sorted(TEST_LINES), sorted(ts.segment_by_lines(TEST_TEXT)))

	def test_segment_by_punctuation(self):
		self.assertEqual(sorted(TEST_MULTIPART_SENTENCES), sorted(ts.segment_by_punctuation(TEST_MULTIPART)))

	def test_segment(self):
		self.assertEqual(sorted(TEST_SENTENCES), sorted(ts.segment(TEST_TEXT)))


TEST_TEXT = """This is a small test text. 

 

Hello! This is a three-part sentence. Is it okay?

"""
TEST_MULTIPART = "Hello! This is a three-part sentence. Is it okay?"
TEST_MULTIPART_SENTENCES = ["Hello!", "This is a three-part sentence.", "Is it okay?"]
TEST_SENTENCES = ["Is it okay?", "This is a small test text.", "Hello!", "This is a three-part sentence."]
TEST_LINES = ['This is a small test text. ', '', ' ', '', 'Hello! This is a three-part sentence. Is it okay?', '']
