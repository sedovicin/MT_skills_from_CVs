import unittest
import categorisators
import json
import dataset_corpus_mgmt as dc_mgmt

class MyTestCase(unittest.TestCase):
	def test_something(self):
		categorisator = categorisators.CategorisatorNN('corpus_train.json')
		with open('../corpus_train.json', 'r', encoding='utf8') as fp:
			x_train = json.load(fp)

		y_train = categorisators.create_y_from_x(x_train, dc_mgmt.import_dataset('../dataset_train.json'))
		x_test = [
			["I'm", "good", "at", "C++"],
			["Marko", "likes", "Linux"],
			["Throw", "me", "a", "ball"],
			["Machine", "learning", "is", "new"]
		]
		y_test = [[1, 0, 0, 1], [0, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0]]

		categorisator.train(x_train, y_train)
		result = categorisator.evaluate(x_test, y_test)
		print(result)
		# categorisator.predict(x_test)


if __name__ == '__main__':
	unittest.main()
