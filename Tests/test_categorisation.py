import unittest
from categorisators import CategorisatorNN
import json


class MyTestCase(unittest.TestCase):
	def test_something(self):

		with open('../corpus_train.json', 'r', encoding='utf8') as fp:
			x_train = json.load(fp)
		y_train = list()
		x_test = [
			["I'm", "good", "at", "C++"],
			["Marko", "likes", "Linux"],
			["Throw", "me", "a", "ball"],
			["Machine", "learning", "is", "new"]
		]
		y_test = [[1, 0, 0, 1], [0, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0]]
		categorisator = CategorisatorNN('corpus_train.json')

		categorisator.train(x_train, y_train)
		result = categorisator.evaluate(x_test, y_test)
		print(result)
		# categorisator.predict(x_test)


if __name__ == '__main__':
	unittest.main()
