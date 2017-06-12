<h1> FastBatch </h1>
<h2> A Multimodal Parallelized Neural Network Iterator for Python 2.7 </h2>

<h3> Description </h3>
<p>A python-based neural network batch iterator with multimodal support.  It works by first splitting the dataset into separate chunks and designating concurrent workers to populate a queue from their assigned chunks. A single process is responsible for retrieving batches from this queue via a get_batch function.  Currently, this version has both numerical and lexical support.</p>

<h3> Requirements </h3>
<ul>
<li> Python 2.7x</li>
<li> numpy </li>
<li> h5py </li>
</ul>

<h3> Data Input Format </h3>
<p> The iterator currently only supports the hdf5 file format. As an input parameter, FastBatch takes in the path to a file list which contains paths to the hdf5 files separated by newline characters. I have provided some functional examples of our input format under the ex folder.
</p>

<h3> Usage </h3>
<p> There are currently two working versions of fastbatch: fastbatch and fastbatch_term.
<li> <b>fastbatch</b>: continues to populate the concurrent queue indefinitely by looping over the file list.</li>
<li><b>fastbatch_term</b>: returns a None tuple after the last element in the file list has been reached.
</p>
Both versions employ the same parameters:
<h4> Mandatory </h4>
<li><b>file_list_name: </b>path to the file list.</li>
<li><b>feat_dsname: </b>the dataset name of the numerical features (ie. openSMILE_features.)</li>
<li><b>word_dsname: </b>the dataset name of the lexical features</li>
<li><b>new_file_list_prefix: </b>prefix for the file list copies that will be generated for each process to work on.</li>
<li><b>k: </b>number of chunks to make. (will assign k workers to work on k chunks of the file list and populate the queue from their respective chunks.)</li>
<li><b>batch_size: </b>number of datapoints per batch.</li>
<li><b>num_timesteps: </b>number of batches desires at a time.</li>

<h4> Optional </h4>
<li><b>shift_step: </b>number of shifts requested for the shifted array (default: 1)</li>
<li><b>word_to_id: </b>Sayan's parameter. (default: None)</li>
<li><b>feat_list: </b>This only works with our modified OpenSmile IS11 features. Leave at None if irrelevant (default: None)</li>
<li><b>remove_sp: </b>removes sp, which are non-word sounds as labeled in the Fisher dataset.  Leave false if irrelevant (default: False) </li>

<h4> Return </h4>
<li><b>orig_word_mat: </b>original word matrix</li>
<li><b>shift_word_mat: </b>shifted word matrix</b></li>
<li><b>feat_mat: </b>feature matrix</li>
<li><b>orig_word_LIWC: </b>LIWC representation based on the original word matrix </li>
<li><b>hdf5_filepath_mat: </b>hdf5 filepaths from which the original word matrix came from. </li>

<h3> Sample Usage</h3>

```python
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
```

<h3> Notes </h3>
Our LIWC parser currently only support single word occurrances, so it isn't the complete LIWC dictionary.  However, it has most of the entries in it.