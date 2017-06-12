import h5py, os, sys, copy, time, traceback
import numpy as np
import multiprocessing
from multiprocessing import Pool
from multiprocessing.managers import BaseManager
from multiprocessing import Queue
from multiprocessing import current_process
from IS11_feat_mapper import search4idx
import liwc_parser_optim as liwc

def clean_line(line):
	return line.replace('\n','').replace('\r','')


class CustomManager(BaseManager): pass
def Manager():
	m = CustomManager()
	m.start()
	return m

class dataset:

	def __init__(self, file_list_name, feat_dsname, word_dsname, word_to_id, remove_sp = False, feat_list = None):
		self.file_list_name = file_list_name
		self.file_list = open(file_list_name, 'r')
		self.word_to_id = word_to_id
		self.curr_hdf5_name	= clean_line(self.file_list.readline())
		self.hdf5 = h5py.File(self.curr_hdf5_name, 'r')
		self.feat_dsname = feat_dsname
		self.word_dsname = word_dsname
		self.iterator_idx = 0
		self.file_pointer = self.file_list.tell()
		self.remove_sp = remove_sp
		self.feat_list = feat_list
		self.liwc_parser = liwc.LIWC_Parser("./LIWC2015_English.dic")
		self.liwc_parser.set_cat2search(['sad', 'anger',  'anx', 'posemo'])
		self.num_liwc_categories = 5


	def get_batch(self, batch_size, num_timesteps, shift_step = 1):
		try:
			words_requested = (batch_size) * (num_timesteps) + shift_step
			feat_size = self.hdf5[self.feat_dsname].shape[-1]
			if self.feat_list is not None: 
				feat_idx = search4idx(self.feat_list)[0]
				feat_size = len(feat_idx)
			feat_mat = np.empty([words_requested, feat_size])
			word_mat = np.array(['                                                  '] * words_requested)
			hdf5_filepath_mat = np.array(['                                                                                                                                                                                                                                                                                                            '] * words_requested)
			words_added = 0
			insertion_idx = 0
			saved_hdf5_filename = self.curr_hdf5_name
			saved_iterator_idx = self.iterator_idx
			saved_file_pointer = self.file_pointer

			already_saved = False
			while words_added < words_requested:
				unstripped_feat = self.hdf5[self.feat_dsname][:] 
				unstripped_words = self.hdf5[self.word_dsname][:]
				if self.feat_list is not None: unstripped_feat = unstripped_feat[:,feat_idx]

				
				if self.remove_sp:
					non_sp_idx = np.where(unstripped_words != 'sp')[0]
					if len(non_sp_idx) == 0: 
						self._open_new_hdf5()
						continue
					if len(non_sp_idx) == len(unstripped_words):
						non_sp_feat = unstripped_feat
						non_sp_words = unstripped_words
					else:
						non_sp_feat = unstripped_feat[non_sp_idx]
						non_sp_words = unstripped_words[non_sp_idx]
					

					feats = np.vstack([non_sp_feat, np.array([-1.0] * feat_size)]) [self.iterator_idx:]
					words = np.append(non_sp_words, np.array(['<eos>'])) [self.iterator_idx:]
				
				else:
					feats = np.vstack([unstripped_feat, np.array([-1.0] * feat_size)]) [self.iterator_idx:]
					words = np.append(unstripped_words, np.array(['<eos>'])) [self.iterator_idx:]
				words_needed = words_requested - words_added
				#print words

				#Needs to open a few file.  Will eat up the entire hdf5
				if len(words) < words_needed:
					word_mat[insertion_idx:insertion_idx + len(words)] = words[0:len(words)]
					feat_mat[insertion_idx:insertion_idx + len(words)] = feats[0:len(words)]
					hdf5_filepath_mat[insertion_idx:insertion_idx + len(words)] = self.curr_hdf5_name

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
					hdf5_filepath_mat[insertion_idx:insertion_idx + len(words)] = self.curr_hdf5_name

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
			hdf5_filepath_mat = hdf5_filepath_mat[:-shift_step]
			feat_mat = feat_mat[:-shift_step].reshape([batch_size, num_timesteps, feat_size])
			# Apply vocabulary on word matrix (original and shifted)
			orig_word_list = list(orig_word_mat)
			shift_word_list = list(shift_word_mat)

			orig_word_text = orig_word_mat.reshape([batch_size, num_timesteps])
			shift_word_text = shift_word_mat.reshape([batch_size, num_timesteps])

			#orig_word_mat = np.array( [ min(self.word_to_id.get(item.lower(), 10000), 9999) for item in orig_word_list] )
			#shift_word_mat = np.array( [ min(self.word_to_id.get(item.lower(), 10000), 9999) for item in shift_word_list] )

			orig_word_mat = orig_word_mat.reshape([batch_size, num_timesteps])
			shift_word_mat = shift_word_mat.reshape([batch_size, num_timesteps])
			hdf5_filepath_mat = hdf5_filepath_mat.reshape([batch_size, num_timesteps])

			orig_word_LIWC = np.zeros([batch_size, num_timesteps, self.num_liwc_categories], dtype=np.float32)
			
			for j in range(batch_size):
				for k in range(num_timesteps):
					liwc_vector = self.liwc_parser.utt2LIWC(   " ".join( orig_word_text[j,:k+1]  ) )
					orig_word_LIWC[j,k,:4] = liwc_vector
					if np.nonzero(liwc_vector)!=0:
						orig_word_LIWC[j,k,4] = 1
			
			#return orig_word_mat, shift_word_mat, feat_mat, orig_word_text, shift_word_text, orig_word_LIWC, hdf5_filepath_mat
			return orig_word_mat, shift_word_mat, feat_mat, orig_word_LIWC, hdf5_filepath_mat
		except Exception as e:
			print 'EXCEPTIONS ON FILE: ' + self.curr_hdf5_name
			print e.args[0]
			sys.stdout.flush()
			exit(-1)


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


def parallel_get_batch(m):

	while(True):
		m['queue'].put(m['dataset'].get_batch(m['batch_size'], m['num_timesteps'], m['shift_step']))

def return_pid(m):
	print(multiprocessing.current_process().name)
	return(m*m)


class p_dataset_iterator:
	def __init__(self, file_list_name, feat_dsname, word_dsname, new_file_list_prefix, k, batch_size, num_timesteps, shift_step = 1, word_to_id = None, feat_list = None, remove_sp = False):
		self.file_list_name = file_list_name
		self.feat_dsname = feat_dsname
		self.word_dsname = word_dsname
		self.word_to_id = word_to_id
		self. new_file_list_prefix = new_file_list_prefix
		self.k = k
		self.batch_size = batch_size
		self.num_timesteps = num_timesteps
		self.shift_step = shift_step
		manager = self._setup_manager()
		self._split_dataset(self.file_list_name, self.new_file_list_prefix, self.k)
		man = multiprocessing.Manager()
		self.dataset_list = list()
		self.q = man.Queue(500)
		for i in range(0, k):
			new_list_path = new_file_list_prefix + str(i).zfill(len(str(k - 1))) + '.txt'
			d = manager.dataset(new_list_path, feat_dsname, word_dsname, self.word_to_id, remove_sp = remove_sp, feat_list = feat_list)
			m = dict({'dataset': d, 'batch_size': batch_size, 'num_timesteps': num_timesteps, 'shift_step': shift_step, 'queue': self.q})
			self.dataset_list.append(m)
		self.pool = Pool(1) # , self.f_init, [self.q]

		self._populate_queue()
		#Preload

	def f_init(self, q):
		parallel_get_batch.q = q

	def get_batch(self):
		return self.q.get()

	def _populate_queue(self):
		# First get process IDs
		self.pool.map_async(return_pid, range(1,10))
		self.pool.map_async(parallel_get_batch, self.dataset_list)

	def _split_dataset(self,file_list_name, new_file_list_prefix, k):
		n = sum(1 for line in open(self.file_list_name))

		if self.k > n:
			print "Don't give me a split size larger than the file itself!"
			exit()

		chunk_size = n/self.k

		file_count = 0
		chunk_count = 0
		chunk_num = 0

		new_filelist = open(self.new_file_list_prefix + str(chunk_count).zfill(len(str(self.k - 1))) + '.txt', 'w')
		orig_file_list = open(self.file_list_name, 'r')
		for line in orig_file_list:
			if file_count == chunk_size and not chunk_count == k - 1 :
				new_filelist.close()
				chunk_count += 1
				file_count = 0
				new_filelist = open(self.new_file_list_prefix + str(chunk_count).zfill(len(str(self.k - 1))) + '.txt', 'w')

			file_count += 1
			new_filelist.write(line)


		new_filelist.close()

	def _setup_manager(self):
		CustomManager.register('dataset', dataset)
		CustomManager.register('queue', Queue)
		manager = Manager()
		return manager

if __name__ == '__main__':
	file_list_name = './ex/file_list.txt'
	feat_dsname = 'openSMILE_features'
	word_dsname = 'words'
	new_file_list_prefix = './file_list_'
	k = 3
	batch_size = 20
	num_timesteps = 20
	shift_step = 1
	pdi = p_dataset_iterator(file_list_name, feat_dsname, word_dsname, new_file_list_prefix, k, batch_size, num_timesteps, shift_step, feat_list = ['f0', 'shimmer', 'jitter', 'voicing', 'rmsenergy'], remove_sp = True)
	
	#pdi.populate_queue()
	time_count = 0
	st = time.time()
	for i in range(0, 100):
		sys.stdout.flush()
		v, w, x ,y, z, a, b = pdi.get_batch()
		if i % 10 == 0:
			ed = time.time()
			print str(i) + 'th batch: ' + str(ed - st)
	ed = time.time()
	print ed - st
	sys.stdout.flush()

