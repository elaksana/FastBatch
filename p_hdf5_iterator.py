import h5py, os, sys, copy, time
import numpy as np
import multiprocessing
from mpi4py import MPI
from multiprocessing import Pool, Queue, cpu_count
from multiprocessing.managers import BaseManager



def clean_line(line):
	return line.replace('\n','').replace('\r','')


def split_dataset(file_list_name, new_file_list_prefix, k):
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

def open_hdf5(filepath):
	h5 = h5py.File(filepath, 'r')
	h5['openSMILE_features'][:]
	h5.close()

if __name__ == '__main__':
	file_list_name = './file_list.txt'
	p = Pool(1)
	a = []
	ifile = open(file_list_name)
	
	for i in range(0,8):
			a.append(clean_line(ifile.readline()))

	st = time.time()
	for i in range(0, 1000):
		p.map(open_hdf5, a)
		
	ed = time.time()
	print (ed - st)