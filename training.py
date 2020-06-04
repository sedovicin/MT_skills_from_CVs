import json
import TextCleaningTool as text_cleaner
import TextSegmentator as segmetator_to_sentences
import SentenceTokenizator as sentence_to_tokens
import POSTagger
import TextNormalizator

dataset = dict()
cleaned_text = text_cleaner.run('cv_extracted/cvs/1_cv.txt')
# text = text_cleaner.extract_from_txt('cv_extracted/cvs/1_cv.txt')
# print("------------TEXT:")
# print(text)
# cleaned_line = text_cleaner.remove_unsupported_chars(text)
# print("------------CLEANED:")
# print(cleaned_line)
line_in_sentences = segmetator_to_sentences.run(cleaned_text)
# print("------------SENTENCES:")
# print(line_in_sentences)
tokens = sentence_to_tokens.run(line_in_sentences)
pos_tags = POSTagger.run(tokens)
# print(pos_tags)

