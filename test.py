from fastbatch import fastbatch
import time, sys, os

if __name__ == '__main__':
	file_list_name = './ex/file_list.txt'
	feat_dsname = 'openSMILE_features'
	word_dsname = 'words'
	new_file_list_prefix = './file_list_'
	k = 3
	batch_size = 20
	num_timesteps = 20
	shift_step = 1
	pdi = fastbatch(file_list_name, feat_dsname, word_dsname, new_file_list_prefix, k, batch_size, num_timesteps, shift_step, feat_list = ['f0', 'shimmer', 'jitter', 'voicing', 'rmsenergy'], remove_sp = True)

	#pdi.populate_queue()
	time_count = 0
	st = time.time()
	for i in range(0, 100):
		sys.stdout.flush()
		a,b,c,d,e = pdi.get_batch()
		if i % 10 == 0:
			ed = time.time()
			print str(i) + 'th batch: ' + str(ed - st)
	ed = time.time()
	print ed - st
	sys.stdout.flush()

