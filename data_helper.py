import re
import logging
import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
import collections as co
import nltk

def clean_str(s):
	stopWordsCalls = ['gesprek','dit','gesprek','kan','voor','kwaliteitsdoeleinden','kwaliteitsdoeleinden','kwaliteits','welkom','geldservice','bij']
	s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
	s = re.sub(r"\'s", " \'s", s)
	s = re.sub(r"\'ve", " \'ve", s)
	s = re.sub(r"n\'t", " n\'t", s)
	s = re.sub(r"\'re", " \'re", s)
	s = re.sub(r"\'d", " \'d", s)
	s = re.sub(r"\'ll", " \'ll", s)
	s = re.sub(r",", " , ", s)
	s = re.sub(r"!", " ! ", s)
	s = re.sub(r"\(", " \( ", s)
	s = re.sub(r"\)", " \) ", s)
	s = re.sub(r"\?", " \? ", s)
	s = re.sub(r"\s{2,}", " ", s)
	s = re.sub(r'\S*(x{2,}|X{2,})\S*',"xxx", s)
	s = re.sub(r'[^\x00-\x7F]+', "", s)
	#s = [str for str in s if not any(i in str for i in stopwords.words("dutch"))]
	#s = [str for str in s if not any(i in str for i in stopWordsCalls)]
	#final_list = []
	#for i in s:
	#	temp = []
	#	for k in i.split(" "):
	#		if not any(i for i in stopwords.words("dutch") if i in k) and not any(i for i in stopWordsCalls if i in k):
	#			temp.append(k)
	#	final_list.append(" ".join(temp))
	#x = final_list
	#words = co.Counter(nltk.corpus.words.words())
	stopWords =co.Counter( nltk.corpus.stopwords.words('dutch') )
	s=[i for i in s if i not in stopWords]
	s=[i for i in s if i not in stopWordsCalls]
	s="".join(s)
	c = co.Counter(s)  
	#s = re.sub(stopwords.words("dutch"),'', s)
	#s = [w for w in s if s not in stopWordsCalls]
	return s.strip().lower()

def load_data_and_labels(filename):
	"""Load sentences and labels"""

	df = pd.read_csv(filename, dtype={'text_from_calls': object})
	selected = ['agent', 'text']
	non_selected = list(set(df.columns) - set(selected))

	df = df.drop(non_selected, axis=1) # Drop non selected columns
	df = df.dropna(axis=0, how='any', subset=selected) # Drop null rows
	df = df.reindex(np.random.permutation(df.index)) # Shuffle the dataframe

	# Map the actual labels to one hot labels
	labels = sorted(list(set(df[selected[0]].tolist())))
	one_hot = np.zeros((len(labels), len(labels)), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	x_raw = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
	y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
	return x_raw, y_raw, df, labels

def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""Iterate the data batch by batch"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(data_size / batch_size) + 1

	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]

if __name__ == '__main__':
	input_file = './dataCalls/allCalls.csv'
	load_data_and_labels(input_file)
