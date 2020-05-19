import pdfminer.high_level as pdf2txt
from pathlib import Path
import os
import re
import docx


def run(file: str):
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

	return remove_unsupported_chars(uncleaned_text)


def extract_from_pdf(pdf_file):
	return pdf2txt.extract_text(pdf_file)


def extract_from_docx(docx_file):
	document = docx.Document(docx_file)
	text = []
	for paragraph in document.paragraphs:
		text.append(paragraph.text)
	return '\n'.join(text)


def extract_from_doc(doc_file):
	# TODO: implement extractFromDoc (is it necessary?)
	raise Exception("Not implemented extract_from_doc")


def extract_from_txt(txt_file):
	with open(txt_file, "r") as file:
		data = file.read()
	return data


def remove_unsupported_chars(text: str):
	return re.sub("[^A-Za-z0-9ŠĐČĆŽšđčćž !\"#$%&/()=?*+\'\n.,<>;:_-]", '', text)