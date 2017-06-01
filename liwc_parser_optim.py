import os, sys, string, time
import numpy as np

def clean_line(line):
	return line.replace('\n','').replace('\r','')

class LIWC_Parser:
	def __init__(self, LIWC_dict_path, get_partial = True):
		self.LIWC_dict_path = LIWC_dict_path
		self.punc = string.punctuation
		self.get_partial = get_partial
		self.initialize_maps()
		self.find_all_cat()

	def find_all_cat(self):
		self.pos_arr = np.array(xrange(0,len(self.id2pos_map))).astype(int)
		self.cat2idx_map = self.full_cat2idx_map
	def get_cat_idx(self):
		return self.cat2idx_map
		
	def set_cat2search(self, cat_arr):
		self.pos_arr = np.array([])
		self.cat2idx_map = {}
		for i in range(0, len(cat_arr)):
			self.pos_arr = np.append(self.pos_arr, self.id2pos_map[self.cat2id_map[cat_arr[i]]])
			self.cat2idx_map[cat_arr[i]] = i
		self.pos_arr = self.pos_arr.astype(int)		
	
	def utt2LIWC(self, utt):
		LIWC_arr = np.zeros(len(self.id2pos_map))

		for word_raw in utt.split():
			word = "".join(c for c in word_raw if c not in self.punc).lower()
			
			 
			try:
				LIWC_arr += self.word2cats_map[word]
			except:
				if not self.get_partial: continue
				for i in range(0,len(word)):
					wordseg = word[0:len(word) - i]
					try:
						LIWC_arr += self.wordseg2cats_map[wordseg]
						break
					except:
						continue				

		return np.array(LIWC_arr).astype(int)[self.pos_arr]

	def get_cat2id_map(self):
		return self.cat2id_map

	def initialize_maps(self):
		LIWC_file = open(self.LIWC_dict_path, 'r')
		LIWC_file.readline()
		cat_pos = 0
		self.id2pos_map, self.cat2idx_map, self.cat2id_map, self.word2cats_map, self.wordseg2cats_map = {}, {}, {}, {}, {}
		for line in LIWC_file:
			line_arr =clean_line(line).split('\t') 
			if line_arr[0] == '%': break
			cat_num = line_arr[0]
			cat_abb = line_arr[1].replace('(', '\t').replace(')', '')
			cat_abb_split = cat_abb.split('\t')
			abb = cat_abb_split[0].replace(' ','')
			cat = cat_abb_split[1]
			self.id2pos_map[int(cat_num)] = cat_pos
			self.cat2id_map[abb] = int(cat_num)
			self.cat2idx_map[abb] = cat_pos
			cat_pos += 1
		self.full_cat2idx_map = self.cat2idx_map
 		for line in LIWC_file:
 			line = clean_line(line)
 			line_split = line.split('\t')
 			word = line_split[0]
 			cats_raw = map(int, line_split[1:])
 			cats = np.zeros(len(self.id2pos_map))
			for c in cats_raw:
				cats[self.id2pos_map[c]] = 1
 			if '*' not in word: self.word2cats_map[word] = cats
 			else: self.wordseg2cats_map[word.replace('*','')] = cats
 			
 		LIWC_file.close()
		
	


if __name__ == '__main__':
	#Please use the dictionary file that I give you.  You will get screwed over otherwise.
	LIWC_dict_path = './LIWC2015_English.dic'
	lp = LIWC_Parser(LIWC_dict_path)
	#If you do not set the cat2search, it will automatically look for every LIWC category.
	#Order of the utterance will depend on the order of categories that you give it.
	
	#lp.set_cat2search(['posemo', 'negemo'])
	lp.set_cat2search(['negemo', 'posemo'])
	lp.find_all_cat()
	st = time.time()
	for i in range(0, 10000):
		lp.utt2LIWC('I hate pigs.  Filled with sensational')
	ed = time.time()
	print (ed - st)
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