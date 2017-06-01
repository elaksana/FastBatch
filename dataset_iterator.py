import h5py, os, sys, copy, time
import numpy as np
from multiprocessing import Pool
from multiprocessing.managers import BaseManager


class CustomManager(BaseManager): pass
def Manager():
	m = CustomManager()
	m.start()
	return m

def clean_line(line):
	return line.replace('\n','').replace('\r','')

#Splits file list into k lists.


def split_dataset(file_list_name, new_file_list_prefix, k):
	n = sum(1 for line in open(file_list_name))
	
	if k > n: 
		print "Don't give me a split size larger than the file itself!"
		exit()

	chunk_size = n/k

	file_count = 0
	chunk_count = 0
	chunk_num = 0

	new_filelist = open(new_file_list_prefix + str(chunk_count).zfill(len(str(k - 1))) + '.txt', 'w')
	orig_file_list = open(file_list_name, 'r') 
	for line in orig_file_list:
		if file_count == chunk_size and not chunk_count == k - 1 :
			new_filelist.close()
			chunk_count += 1
			file_count = 0	
			new_filelist = open(new_file_list_prefix + str(chunk_count).zfill(len(str(k - 1))) + '.txt', 'w')

		file_count += 1
		new_filelist.write(line)
		
	
	new_filelist.close()



class dataset:
	
	def __init__(self, file_list_name, feat_dsname, word_dsname):
		self.file_list_name = file_list_name
		self.file_list = open(file_list_name, 'r')
		self.curr_hdf5_name	= clean_line(self.file_list.readline())	
		self.hdf5 = h5py.File(self.curr_hdf5_name, 'r')
		self.feat_dsname = feat_dsname
		self.word_dsname = word_dsname
		self.iterator_idx = 0
		self.file_pointer = self.file_list.tell()
		

	def get_batch(self, batch_size, num_timesteps, shift_step = 1):
		words_requested = (batch_size) * (num_timesteps) + shift_step
		feat_size = self.hdf5[self.feat_dsname].shape[-1]
		feat_mat = np.empty([words_requested, feat_size])
		word_mat = np.array(['                                                  '] * words_requested)

		words_added = 0
		insertion_idx = 0
		
		saved_hdf5_filename = self.curr_hdf5_name
		saved_iterator_idx = self.iterator_idx
		saved_file_pointer = self.file_pointer
		
		already_saved = False
		time.sleep(0.5)
		while words_added < words_requested:
			feats = np.vstack([self.hdf5[self.feat_dsname], np.array([-1.0] * feat_size)]) [self.iterator_idx:]
			words = np.append(self.hdf5[self.word_dsname], np.array(['<eos>'])) [self.iterator_idx:]

			words_needed = words_requested - words_added
			
			#Needs to open a few file.  Will eat up the entire hdf5
			if len(words) < words_needed:
				word_mat[insertion_idx:insertion_idx + len(words)] = words[0:len(words)]
				feat_mat[insertion_idx:insertion_idx + len(words)] = feats[0:len(words)]
				
				words_added += len(words)
				insertion_idx += len(words)
				if words_added >= (words_requested - shift_step) and not already_saved:
					exceed = len(words) - (words_added - (words_requested - shift_step))
					saved_iterator_idx = self.iterator_idx + exceed
					saved_hdf5_filename = self.curr_hdf5_name
					saved_file_pointer = self.file_pointer
					already_saved = True
				
				self._open_new_hdf5()


			#Middle of the hdf5 file.
			else:
				feat_mat[insertion_idx:insertion_idx + words_needed] = feats[:words_needed]
				word_mat[insertion_idx:insertion_idx + words_needed] = words[:words_needed]

				words_added += words_needed
				insertion_idx += words_needed
				
				if words_added >= (words_requested - shift_step) and not already_saved:
					exceed = words_needed - shift_step 
					saved_hdf5_filename = self.curr_hdf5_name
					saved_iterator_idx = self.iterator_idx + exceed
					saved_file_pointer = self.file_pointer
					already_saved = True
				
				self.iterator_idx += words_needed
				

		if self.curr_hdf5_name != saved_hdf5_filename:
			self.hdf5.close()
			self.curr_hdf5_name = saved_hdf5_filename
			self.hdf5 = h5py.File(self.curr_hdf5_name, 'r')
			self.file_pointer = saved_file_pointer
			self.file_list.seek(self.file_pointer)
		self.iterator_idx = saved_iterator_idx
		

		orig_word_mat = word_mat[:-shift_step]
		shift_word_mat = word_mat[shift_step:]
		
		feat_mat = feat_mat[:-shift_step].reshape([batch_size, num_timesteps, feat_size])
		orig_word_mat = orig_word_mat.reshape([batch_size, num_timesteps])
		shift_word_mat = shift_word_mat.reshape([batch_size, num_timesteps])
		
		return orig_word_mat, shift_word_mat, feat_mat




	def _open_new_hdf5(self):
		self.hdf5.close()
		self.curr_hdf5_name = clean_line(self.file_list.readline())
		if self.curr_hdf5_name == '':
			self._reset_list()
			self.curr_hdf5_name = clean_line(self.file_list.readline())
		self.hdf5 = h5py.File(self.curr_hdf5_name, 'r')
		self.iterator_idx = 0
		self.file_pointer = self.file_list.tell()

	
	
	def _reset_list(self):
		self.file_list.seek(0)
	

def setup_manager():
	CustomManager.register('dataset', dataset)
	manager = Manager()
	return manager



if __name__ == '__main__':
	file_list_name = './file_list.txt'
	feat_dsname = 'openSMILE_features'
	word_dsname = 'words'
	new_file_list_prefix = './file_list'
	batch_size = 20
	num_timesteps = 15
	shift_step = 1


	d = dataset(file_list_name, feat_dsname, word_dsname)

	st = time.time()
	for i in range(0, 100):
		d.get_batch(batch_size, num_timesteps, shift_step)[0]
	ed = time.time()
	print 'Total extraction time: ' + str(float(ed - st))
	'''
	split_dataset(file_list_name, new_file_list_prefix + '_', k)

	manager = setup_manager()

	dataset_list = []
	for i in range(0, k):
		new_list_path = new_file_list_prefix + '_' + str(i).zfill(len(str(k - 1))) + '.txt'
		d = manager.dataset(new_list_path, feat_dsname, word_dsname)
		m = {'dataset': d, 'batch_size': batch_size, 'num_timesteps': num_timesteps, 'shift_step': shift_step}
		dataset_list.append(m)
	print len(dataset_list)
	#exit()
	#dataset_list = [1,2,3]
	#exit()
	p = Pool(k)

	print p.map(parallel_get_batch, dataset_list)[0][0]
	print p.map(parallel_get_batch, dataset_list)[0][0]
	#print p.map(parallel_get_batch, dataset_list)
	'''
