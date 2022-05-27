import json
import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf_vec = TfidfVectorizer(max_df=0.8, min_df=0.1, stop_words=stop_words) 

def get_label_embedding(label_dict, input_dir, output_dir):
    :param label_dict: label dictionary, the same order with model  
    :param input_dir: input dataset files  
    :param output_dir: output json files  

	label_list = []
	with open(label_dict) as f:
	    for line in f.readlines():
	        line = line.rstrip().split('\t')
	        label_list.append(line[0])
	label_samples = {}
	label_embedding = {}
	with open(input_dir) as f:
		for line in f.readlines():
			sample = json.loads(line.rstrip())
			for label in sample["label"]:
				if label not in label_samples:
					label_samples[label] = [" ".join(sample["token"])]
				else:
					label_samples[label].append(" ".join(sample["token"]))
	# tfidf method
	for label in label_list:
		if label not in label_samples:
			word = label.split(' ')
		else:
			corpus = label_samples[label]
			#count = vectorizer.fit_transform(corpus)
			#tfidf_matrix = transformer.fit_transform(count)
			tfidf_matrix = tfidf_vec.fit(corpus)
			sorted(tfidf_matrix.vocabulary_, key=lambda x:x[1], reverse=True)
			print("label " + str(label) + ": " + str(tfidf_matrix.vocabulary_.keys()))
			print("len:", len(tfidf_vec.vocabulary_.keys()))
			word = list(tfidf_matrix.vocabulary_.keys())
		label_embedding[label] = word
	with open(output_dir, 'w') as f:
		for label in label_list:
			instance = {"label": [label], "token": label_embedding[label]}
			json.dump(instance,f)
			f.write("\n")

get_label_embedding("./vocab_wos/label.dict", "./data/wos_train.json", "./data/wos_label_desc.json")