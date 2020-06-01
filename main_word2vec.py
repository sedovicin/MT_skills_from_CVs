import vectorizators

word2vec = vectorizators.word2vec_vectorizator()
print(word2vec.wv.most_similar('love'))

