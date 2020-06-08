import pdfminer.high_level as pdf2txt
from pathlib import Path
import os
import re
import docx


def extract(file: str):
	"""
	Extracts text from given file.
	Currently supports PDF, DOCX and TXT files.

	:param file: path to file to be processed
	:type file: str
	:return: All the text from file as one string
	:rtype: str
	"""
	# TODO: check file type instead of file extension
	extension = Path(file).suffix.lower()
	if not os.path.isfile(file):
		raise FileNotFoundError()
	if extension == '.pdf':
		uncleaned_text = extract_from_pdf(file)
	elif extension == '.docx':
		uncleaned_text = extract_from_docx(file)
	elif extension == '.doc':
		uncleaned_text = extract_from_doc(file)
	elif extension == '.txt':
		uncleaned_text = extract_from_txt(file)
	else:
		raise TypeError("Unsupported file type!")

	return uncleaned_text


def extract_from_pdf(pdf_file):
	"""
	Extracts text from PDF file.

	:param pdf_file: path to file to be processed
	:type pdf_file: str
	:return: All the text from file as one string
	:rtype: str
	"""
	return pdf2txt.extract_text(pdf_file)


def extract_from_docx(docx_file):
	"""
	Extracts text from DOCX file.

	:param docx_file: path to file to be processed
	:type docx_file: str
	:return: All the text from file as one string
	:rtype: str
	"""
	document = docx.Document(docx_file)
	text = []
	for paragraph in document.paragraphs:
		text.append(paragraph.text)
	return '\n'.join(text)


def extract_from_doc(doc_file):
	# TODO: implement extractFromDoc (is it necessary?)
	raise Exception("Not implemented extract_from_doc")


def extract_from_txt(txt_file):
	"""
		Extracts text from TXT file. Supports only files encoded with UTF8.

		:param txt_file: path to file to be processed
		:type txt_file: str
		:return: All the text from file as one string
		:rtype: str
		"""
	with open(txt_file, "r", encoding='utf8') as file:
		data = file.read()
	return data


def remove_unsupported_chars(text: str):
	"""
	Removes characters that are usually not present in texts, like non-typical UTF8 characters.
	This method is adjusted for english and croatian language only.

	:param text: text to be processed
	:type text: str
	:return: new string without the removed characters
	:rtype: str
	"""
	return re.sub("[^A-Za-z0-9ŠĐČĆŽšđčćž !\"#$%&/()=?*+\'\n.,<>;:_-]", '', text)
