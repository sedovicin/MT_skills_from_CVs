import pdfminer.high_level as pdf2txt;
from pathlib import Path;
import os;
import re;


def run(file: str):
	# TODO: check file type instead of file extension
	extension = Path(file).suffix.lower();
	if (os.path.isfile(file) == False):
		raise FileNotFoundError();
	if (extension == '.pdf'):
		uncleanedText = extractFromPDF(file);
	elif (extension == '.docx'):
		uncleanedText = extractFromDocx(file);
	elif (extension == '.doc'):
		uncleanedText = extractFromDoc(file);
	elif (extension == '.txt'):
		uncleanedText = extractFromTxt(file);
	else:
		raise TypeError("Unsupported file type!");

	return removeUnsupportedChars(uncleanedText);


def extractFromPDF(pdfFile):
	return pdf2txt.extract_text(pdfFile);


def extractFromDocx(docxFile):
	raise Exception("Not implemented extractFromDocx");


def extractFromDoc(docFile):
	raise Exception("Not implemented extractFromDoc");


def extractFromTxt(txtFile):
	raise Exception("Not implemented extractFromTxt");


def removeUnsupportedChars(text: str):
	return re.sub('[^A-Za-z0-9ŠĐČĆŽšđčćž !\"#$%&/()=?*+\'\\\n\.,<>;:\-\_]', '', text);