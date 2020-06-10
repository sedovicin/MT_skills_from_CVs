import json
import os.path as path

import categorisators
import dataset_corpus_mgmt as dc_mgmt


def main():
	if (not path.isfile('corpus_train.json')) and (not path.isfile('dataset_train.json')):
		dc_mgmt.create_dataset_corpus('train', 0, 10000)
	if (not path.isfile('corpus_test.json')) and (not path.isfile('dataset_test.json')):
		dc_mgmt.create_dataset_corpus('test', 10000, 10594)

	categorisator = categorisators.CategorisatorNN('corpus_train.json')
	with open('corpus_train.json', 'r', encoding='utf8') as fp:
		x_train = json.load(fp)
	y_train = categorisators.create_y_from_x(x_train, dc_mgmt.import_dataset('dataset_train.json'))

	with open('corpus_test.json', 'r', encoding='utf8') as fp:
		x_test = json.load(fp)
	y_test = categorisators.create_y_from_x(x_test, dc_mgmt.import_dataset('dataset_test.json'))

	categorisator.train(x_train, y_train, batch_size=1000, epochs=5)
	result = categorisator.evaluate(x_test, y_test)
	print("Loss, accuracy:")
	print(result)


if __name__ == "__main__":
	main()
