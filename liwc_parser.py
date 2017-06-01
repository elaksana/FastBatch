import os, sys, string, time
import numpy as np

def clean_line(line):
	return line.replace('\n','').replace('\r','')

class LIWC_Parser:
	def __init__(self, LIWC_dict_path):
		self.LIWC_dict_path = LIWC_dict_path
		self.punc = string.punctuation
		'''
		initiialize:
			1. id2cat
			2. cat2abb
			3. word2cats

		'''
		self.initialize_maps()
		self.find_all_cat()
		self.cat_idx_mapping = {}
	def find_all_cat(self):
		self.cat_arr = self.full_cats

	def get_idx_cat(self):
		self.cat_idx_mapping = {}
		for i in range(0, len(self.cat_arr)):
			self.cat_idx_mapping[i] = self.cat_arr[i]
		return self.cat_idx_mapping
	def set_cat2search(self, cat_arr):
		for i in range(0, len(cat_arr)):
			if cat_arr[i] not in self.cat2abb_map.values(): cat_arr[i] = self.cat2abb_map[cat_arr[i]]
		self.cat_arr = cat_arr
		
	
	def utt2LIWC(self, utt):
		LIWC_arr = [0] * len(self.cat_arr)

		#print self.cat2id_map
		#exit()
		for word_raw in utt.split():
			word = "".join(c for c in word_raw if c not in self.punc).lower()
			
			for i in range(0, len(self.cat_arr)):
				cat_id = self.cat2id_map[self.cat_arr[i]]
				try:
					if cat_id in self.word2cats_map[word]: 
						#print 'word' + '/' + str(word) + '/' + str(cat_id)
						LIWC_arr[i] += 1
				except:
					#if self.cat2id_map[self.cat_arr[i]]
					for wordseg in self.wordseg2cats_map:
						
						if word.startswith(wordseg):
							if cat_id in self.wordseg2cats_map[wordseg]: 
								#print 'seg' + '/' + str(word) + '/' + str(cat_id)
								LIWC_arr[i] += 1

		return np.array(LIWC_arr)
	def get_cat2abb_mapping(self):
		return self.cat2abb_map

	def get_cat2id_mapping(self):
		return self.cat2id_map

	def initialize_maps(self):
		LIWC_file = open(self.LIWC_dict_path, 'r')
		LIWC_file.readline()
		self.cat2id_map, self.cat2abb_map, self.word2cats_map, self.wordseg2cats_map, self.full_cats = {}, {}, {}, {}, []
		for line in LIWC_file:
			line_arr =clean_line(line).split('\t') 
			if line_arr[0] == '%': break
			cat_num = line_arr[0]
			cat_abb = line_arr[1].replace('(', '\t').replace(')', '')
			cat_abb_split = cat_abb.split('\t')
			abb = cat_abb_split[0].replace(' ','')
			cat = cat_abb_split[1]
			self.cat2id_map[abb] = int(cat_num)
			self.cat2abb_map[cat] = abb
			self.full_cats.append(abb)
			
 		for line in LIWC_file:
 			line = clean_line(line)
 			line_split = line.split('\t')
 			word = line_split[0]
 			cats = map(int, line_split[1:])
 			if '*' not in word: self.word2cats_map[word] = cats
 			else: self.wordseg2cats_map[word.replace('*','')] = cats
 			#print map(int, line_split[1:])
 		LIWC_file.close()
		
	


if __name__ == '__main__':
	#Please use the dictionary file that I give you.  You will get screwed over otherwise.
	LIWC_dict_path = './LIWC2015_English.dic'
	lp = LIWC_Parser(LIWC_dict_path)
	#If you do not set the cat2search, it will automatically look for every LIWC category.
	#Order of the utterance will depend on the order of categories that you give it.
	lp.set_cat2search(['posemo', 'negemo'])
	lp.find_all_cat()
	st = time.time()
	for i in range(0, 10):
		print lp.utt2LIWC('hello! My name is Eugene.  This food tastes heavenly.  The paper writing process hurts very badly.')
	ed = time.time()
	print (ed - st)
	exit()
	'''
	print lp.get_idx_cat()
	#Use all the LIWC categories.
	lp.find_all_cat()
	print lp.get_idx_cat()
	#Order of the full category option is the original LIWC category order.
	print lp.utt2LIWC('hello! My name is Eugene.  This food tastes heavenly.  The paper writing process hurts.')
	#Set back to posemo/negemo, but switch orders this time around.
	lp.set_cat2search(['negemo', 'posemo'])
	print lp.utt2LIWC('hello! My name is Eugene.  This food tastes heavenly.  The paper writing process hurts very badly.')
	print lp.get_idx_cat()
	'''