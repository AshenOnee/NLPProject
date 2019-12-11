import nltk
from pymystem3 import Mystem
from nltk import bigrams
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords

corpus_root = 'Отзывы'
corpus_negative = PlaintextCorpusReader(corpus_root + '/Отрицательные', '.*')
corpus_positive = PlaintextCorpusReader(corpus_root + '/Положительные', '.*')

print('Корпус с негативными отзывами:')
print(corpus_negative.fileids())
print(corpus_negative.words('1.txt'))

print()
print('Корпус с позитивными отзывами:')
print(corpus_positive.fileids())
print(corpus_positive.words('1.txt'))

############################### Загрузка из файла ###############################
f = open('Отзывы/Положительные.txt', encoding="utf-8")
positive_raw = f.read()
tokens = nltk.word_tokenize(positive_raw)
positive = nltk.Text(tokens)

f = open('Отзывы/Отрицательные.txt', encoding="utf-8")
negative_raw = f.read()
tokens = nltk.word_tokenize(negative_raw)
negative = nltk.Text(tokens)

########################## Мера лексического разнообразия #######################
tokens = [str(token).lower() for token in positive.tokens if str(token).isalpha()]
count_unique_words = len(set(tokens))
count_total_words = len(tokens)
print('\n')
print('Уникальных слов в положительных отзывах: ' + str(count_unique_words))
print('Всего слов в положительных отзывах: ' + str(count_total_words))
print('Мера лексического разнообразия в положительных отзывах: ' + str(count_unique_words / count_total_words))

tokens = [str(token).lower() for token in negative.tokens if str(token).isalpha()]
count_unique_words = len(set(tokens))
count_total_words = len(tokens)
print()
print('Уникальных слов в негативных отзывах: ' + str(count_unique_words))
print('Всего слов в негативных отзывах: ' + str(count_total_words))
print('Мера лексического разнообразия в негативных отзывах: ' + str(count_unique_words / count_total_words))

############### Мера лексического разнообразия (без стоп-слов) ##################
russian_stopwords = stopwords.words("russian")

tokens = [str(token).lower() for token in positive.tokens if str(token).isalpha()]
tokens_without_stopwords = [word for word in tokens if word not in russian_stopwords]
count_unique_words = len(set(tokens_without_stopwords))
count_total_words = len(tokens_without_stopwords)
print()
print('Уникальных слов в положительных отзывах (без стоп слов): ' + str(count_unique_words))
print('Всего слов в положительных отзывах (без стоп слов): ' + str(count_total_words))
print('Мера лексического разнообразия в положительных отзывах (без стоп-слов): '
      + str(count_unique_words / count_total_words))

tokens = [str(token).lower() for token in negative.tokens if str(token).isalpha()]
tokens_without_stopwords = [word for word in tokens if word not in russian_stopwords]
count_unique_words = len(set(tokens_without_stopwords))
count_total_words = len(tokens_without_stopwords)
print()
print('Уникальных слов в негативных отзывах (без стоп слов): ' + str(count_unique_words))
print('Всего слов в негативных отзывах (без стоп слов): ' + str(count_total_words))
print('Мера лексического разнообразия в негативных отзывах (без стоп-слов): '
      + str(count_unique_words / count_total_words))

############### Анализ распределения частот наиболее распространенных слов ##################
mystem = Mystem()

positive_lemmas_tokens = nltk.word_tokenize("".join(mystem.lemmatize(positive_raw)))
negative_lemmas_tokens = nltk.word_tokenize("".join(mystem.lemmatize(negative_raw)))

tokens = [str(token).lower() for token in positive_lemmas_tokens if str(token).isalpha()]
freq_dist = nltk.FreqDist(word for word in tokens)
print('\n')
print('Распределение частот 10-ти наиболее распространенных слов положительных отзывов')
print(freq_dist.most_common(10))
freq_dist.plot(10, cumulative=True)

tokens = [str(token).lower() for token in negative_lemmas_tokens if str(token).isalpha()]
freq_dist = nltk.FreqDist(word for word in tokens)
print()
print('Распределение частот 10-ти наиболее распространенных слов отрицательных отзывов')
print(freq_dist.most_common(10))
freq_dist.plot(10, cumulative=True)

### Анализ распределения частот наиболее распространенных слов (без стоп-слов)###
tokens = [str(token).lower() for token in positive_lemmas_tokens if
          str(token).isalpha() and str(token).lower() not in russian_stopwords]
freq_dist = nltk.FreqDist(word for word in tokens)
print()
print('Распределение частот 10-ти наиболее распространенных слов положительных отзывов (без стоп-слов)')
print(freq_dist.most_common(10))
freq_dist.plot(10, cumulative=True)

tokens = [str(token).lower() for token in negative_lemmas_tokens if
          str(token).isalpha() and str(token).lower() not in russian_stopwords]
freq_dist = nltk.FreqDist(word for word in tokens)
print()
print('Распределение частот 10-ти наиболее распространенных слов отрицательных отзывов (без стоп-слов)')
print(freq_dist.most_common(10))
freq_dist.plot(10, cumulative=True)

################################ Отбор слов по их длине ##############################
minimum_word_length = 10
tokens = [str(token).lower() for token in set(positive.tokens) if
          str(token).isalpha() and len(token) >= minimum_word_length]
print('\n')
print('Слова из положительных отзывов длина которых не меньше ' + str(minimum_word_length))
print(tokens[:20])

tokens = [str(token).lower() for token in set(negative.tokens) if
          str(token).isalpha() and len(token) >= minimum_word_length]
print()
print('Слова из негативных отзывов длина которых не меньше ' + str(minimum_word_length))
print(tokens[:20])

######################### Отбор слов по их длине (без стоп-слов) ######################
tokens = [str(token).lower() for token in set(positive.tokens) if
          str(token).isalpha() and len(token) >= minimum_word_length and str(token).lower() not in russian_stopwords]
print()
print('Слова из положительных отзывов длина которых не меньше ' + str(minimum_word_length) + " (без стоп-слов)")
print(tokens[:20])

tokens = [str(token).lower() for token in set(negative.tokens) if
          str(token).isalpha() and len(token) >= minimum_word_length and str(token).lower() not in russian_stopwords]
print()
print('Слова из негативных отзывов длина которых не меньше ' + str(minimum_word_length) + " (без стоп-слов)")
print(tokens[:20])

######### Анализ распределения частот наиболее распространенных биграмм ##############
tokens = [str(token).lower() for token in positive.tokens if str(token).isalpha()]
token_bigrams = bigrams(tokens)
freq_dist = nltk.FreqDist(bigram for bigram in token_bigrams)
print('\n')
print('Распределение частот 10-ти наиболее распространенных биграмм положительных отзывов')
print(freq_dist.most_common(10))

tokens = [str(token).lower() for token in negative.tokens if str(token).isalpha()]
token_bigrams = bigrams(tokens)
freq_dist = nltk.FreqDist(bigram for bigram in token_bigrams)
print()
print('Распределение частот 10-ти наиболее распространенных биграмм отрицательных отзывов')
print(freq_dist.most_common(10))

#### Анализ распределения частот наиболее распространенных биграмм (без стоп-слов)####
tokens = [str(token).lower() for token in positive.tokens if
          str(token).isalpha() and str(token).lower() not in russian_stopwords]
token_bigrams = bigrams(tokens)
freq_dist = nltk.FreqDist(bigram for bigram in token_bigrams)
print()
print('Распределение частот 10-ти наиболее распространенных биграмм положительных отзывов (без стоп-слов)')
print(freq_dist.most_common(10))

tokens = [str(token).lower() for token in negative.tokens if
          str(token).isalpha() and str(token).lower() not in russian_stopwords]
token_bigrams = bigrams(tokens)
freq_dist = nltk.FreqDist(bigram for bigram in token_bigrams)
print()
print('Распределение частот 10-ти наиболее распространенных биграмм отрицательных отзывов (без стоп-слов)')
print(freq_dist.most_common(10))
