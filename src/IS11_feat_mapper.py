import os
import numpy as np

IS11_feat_map_path = './IS11_feat_names.txt'
feat2idx = {}

def load_map():
	idx = 0
	for line in open(IS11_feat_map_path, 'r'):
		line = line.replace('\n','').replace('\r','').replace('@attribute','').lower()
		feat2idx[line] = idx
		idx += 1

def search4idx(feat_name_list):
	idx_arr = np.array([]).astype(int)
	featname_arr = np.array([]).astype(str)
	for key in feat2idx:
		for feat_name in feat_name_list:
			if feat_name in key:
				feat_name = feat_name.lower()
				idx_arr = np.append(idx_arr, feat2idx[key])
				featname_arr = np.append(featname_arr, key)
	sorted_idx_ord = np.argsort(idx_arr)
	return idx_arr[sorted_idx_ord], featname_arr[sorted_idx_ord]


load_map()