# Jian Hu is writing the code here

# ''' Day4 Exercise for Natural Language Processing '''

# import the methods and packages we need
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from string import digits, punctuation
import nltk
import string
from collections import Counter


# 1. Regular Expressions###
# a) solve the first eight regex practice problems
# 1. [^720p]
# 2. ([0-9]{3})
# 3. ^([\w\.]*)
# 4. <(\w+)
# 5. (\w+)\.(jpg|png|gif)$
# 6. ^\s*(.*)\s*$
# 7. (\w+)\(([\w\.]+):(\d+)\)
# 8. Protocol: (\w+)://
#    Host: ://([\w\-\.]+)
#    Port: (:(\d+))


# 2. Speeches I###

# a). read up about path.lib.Path().glob()， only that start with "R0", encoded in UTF8.
files = sorted(Path('./data/speeches').glob('R0*'))
print(files)
corpus = []
for name in files:
    try:
        f = open(name, "r", encoding="utf-8")
        corpus.append(f.readlines()[0])
        f.close()
    except UnicodeDecodeError:
        print(name)

# b). Vectorize he speeches using tfidf using 1-grams, 2-grams and 3-grams while removing English stopwords
_stemmer = nltk.snowball.SnowballStemmer('english')


def tokenize_and_stem(text):
    """Return tokens of text deprived of numbers and interpunctuation."""
    text = text.translate(str.maketrans({p: "" for p in digits + punctuation}))
    return [_stemmer.stem(t) for t in nltk.word_tokenize(text.lower())]


_stopwords = nltk.corpus.stopwords.words('english')
_stopstring = " ".join(_stopwords)
_stopwords = tokenize_and_stem(_stopstring)

# tfdif matrix
tfidf = TfidfVectorizer(stop_words=_stopwords, tokenizer=tokenize_and_stem, ngram_range=(1, 3))
tfidf_matrix = tfidf.fit_transform(corpus)
print(tfidf.vocabulary_)
tfidf_matrix.todense()

# get DataFrame
terms = tfidf.get_feature_names()
df = pd.DataFrame(tfidf_matrix.todense().T, index=terms)
cols = list(df.columns.values)
index = 1
for column in cols:
    cols[index - 1] = "Data" + str(index)
    index += 1
vals = df.values.tolist()
df_tfidf = pd.DataFrame(vals,   columns=cols)
print(df_tfidf)


# c). pickle the resulting sparse matrix
file = open("./output/speech_matrix.pk", "wb")
pickle.dump(tfidf_matrix, file)
file.close()

# save the terms as well as ./output/terms.csv
df_terms = pd.DataFrame(terms, columns=["terms"])
df_terms.to_csv("./output/terms.csv", sep=",")


# 3. Speeches II###

# a) read the count-matrix using pickle.load()
file = open("./output/speech_matrix.pk", "rb")
speeches_file = pickle.load(file)


# b) Using the matrix, create a dengrogram of hierarchichal clustering
speeches_file_dens = speeches_file.toarray()
clustering = AgglomerativeClustering().fit(speeches_file_dens)
linkage_matrix = linkage(speeches_file_dens, metric="cosine", method="complete")
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix)

# remove the x-ticks
plt.tick_params(
    axis='x',           # changes apply to the x-axis
    which='both',       # both major and minor ticks are affected
    bottom=False,       # ticks along the bottom edge are off
    top=False,          # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are of
plt.show()


# c). Save the dendrogram as pdf
plt.savefig("./output/speeches_dendrogram.pdf")


# 4. Job Ads

# a) read the text file and parse the lines to get DataFrame with three columns
lists = []
with open("./data/Stellenanzeigen.txt", "r", encoding='utf-8') as stelle_file:
    for line in stelle_file:
        line = line.strip("\n")
        if line != '':
            lists.append(line)

# the third and fourth line are saved as different ads and we are saving them in open variable
third = lists[3] + lists[4]
lists.pop(4)
lists[3] = third

# tokenize and remove stopwords from the advertisements
_stemmer = nltk.snowball.SnowballStemmer('german')


def tokenize_and_stem(text):
    """Return tokens of text deprived of numbers and interpunctuation."""
    text = text.translate(str.maketrans({p: "" for p in digits + punctuation}))
    return [_stemmer.stem(t) for t in nltk.word_tokenize(text.lower())]


_stopwords = nltk.corpus.stopwords.words('german')
_stopstring = " ".join(_stopwords)
_stopwords = tokenize_and_stem(_stopstring)

# sometimes the tokens are just characters and we want to remove them as well as they dont make sense
excluded = [x for x in string.ascii_lowercase]

# divide the lists between the two
ads = []
newsp = []
for num, i in enumerate(lists):
    if num % 2 == 0:
        newsp.append(i)
    else:  # run all of the text cleaning stuff on advertisements
        i = re.sub(r'[^\w\s]', '', i)
        i = tokenize_and_stem(i)
        i = [w for w in i if w not in _stopwords]
        i = [w for w in i if w not in excluded]
        ads.append(i)

# get date and newspaper in different variables
tdate = [i.split(',', 1)[1] for i in newsp]
newspaper = [i.split(',', 1)[0] for i in newsp]

# create the DataFrame
d = {'Newspaper': newspaper, 'Date': tdate, 'Job Ad': ads}
df_ads = pd.DataFrame(data=d)

# replace März by 3.
df_ads['Date'] = df_ads['Date'].str.replace("März", '3.')
df_ads = df_ads.astype({'Date': 'datetime64[ns]'})

# get the year
df_ads['Year'] = df_ads['Date'].dt.year


# b) Create a new column counting the number of words per job ad.
# add a word count column
df_ads['Word Count'] = df_ads.apply(lambda row: len(row['Job Ad']), axis=1)

# plot the average job ad length by decade
# bar chart
# create means to plot
MeanWC = pd.DataFrame(df_ads.groupby([(df_ads.Year//10*10)])["Word Count"].mean())
MeanWC.plot(kind='bar', legend=None)
plt.xlabel('Decades')
plt.ylabel('Average number of words advertised')

# line plot
MeanWC.plot(legend=None)
plt.xlabel('Decades')
plt.ylabel('Average number of words advertised')


# c) create a second DataFrame that aggregates the job ads by decade
# create a new dataset and use this function to append couple of rows for the same decade to each other
newdf = pd.DataFrame(df_ads[['Year', 'Job Ad']].groupby([(df_ads.Year // 10 * 10)]).aggregate(lambda x: list(x)))

# The above gives us a list of list for terms used and we have to create a flatten list
# so we create this function
# to flatten a list


def flatten(t):
    return [item for sublist in t for item in sublist]


# now used it on rows to flatten the lists of list in Job Ad
newdf['Job Ad'] = newdf['Job Ad'].transform(lambda x: flatten(x))

# the most common five words per decade
newdf['Most Common Decade'] = newdf.apply(lambda row: Counter(row['Job Ad']).most_common(5), axis=1)

# "zurich", "gesucht", "offert" and "sofort" are the most common words across decades.

# the most common words for Decade 1900 are "Eintritt" and "offert" with 6 times

# the most common words for Decade 1910 are "gesucht", "tuchtig", "offert" with 6 times

# the most common word for Decade 1920 is  "tuchtig" with 6 times.

# the most common word for Decade 1930 is  "offert" with 7 times

# the most common word for Decade 1940 is  "chiffr" with 9 times.

# the most common word for Decade 1950 is  "zurich" with 8 times.

# the most common word for Decade 1960 is  "zurich" with 10 times.

# the most common word for Decade 1970 is  "aufgab" with 10 times.

# the most common word for Decade 1980 is  "zurich" with 8 times.

# the most common word for Decade 1990 is  "zurich" with 13 times.

# the most common words for Decade 2000 are "the" with 18 times and "sowi" with 14 times.
