# Unintentional Fragments Annotation

## Introduction
###
In LC-MS dataset, features of fragments are generated from fragmentation process which is the dissociation of energetically unstable molecular ions and is generally not a desired effect.
As a result, signals from unique features are often confounded with unintentional fragments.
This algorithm firstly uses non-negative LASSO regression model to generate a basis set which contains linearly independent features (each feature represents a unique group) and then use the generated basis set to decompose other features since fragment features are often generated from one or multiple precursor ions. 
Based on coefficients of decomposition results, one feature could be assigned to one or multiple groups. After assigning all the features, each unique group could be regarded as a reconstructed spectrum which could be compared against spectra in databases for best match.
In our algorithm, we compare cosine similarity between two spectra by defining inner product as integral of multiplication of two Gaussian distributions, 
which could successfully annotate thousands of features including fragments, common adducts, isotopes and dimers in MS1 dataset with the help
of MassBank of North America (MoNA) database.
## Package dependency
###
python packages:
```python
import numpy as np
import pandas as pd
import cvxopt
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from tqdm.auto import tqdm
from dataclasses import dataclass, field, asdict
import copy
import json
import pickle as pkl
import warnings
import math
import typing
import datetime
```
R libraries:
```R
library('xcms')
```

## Usage
### step 1: data preparation
####
- Download experimental LC-MS/MS dataset (both positive and negative modes) at https://mona.fiehnlab.ucdavis.edu/downloads
- Process the .json files using 'MoNA_data_filter.py' file to only extract the needed spectra information
- Use MS-Convert (http://proteowizard.sourceforge.net/tools.shtml) to convert vendor formatted LC-MS data into .mzML files.
- In order to detect all the peaks and extract them from .mzML files, run 'generate_features.R' file. 
This step would output three .txt files (data_ints.txt, data_mzs.txt and data_rts.txt) and one peak_info.csv file. Then put
these four files into one folder. 

### step 2: features matching against database
####
After generating four files mentioned in previous step, run 'ms_data_workflow.py' to generate matching results of the MS1 dataset. 
Details for each step of our algorithm could be found in the
notes of 'ms_data_workflow.py'.

## Output
The output result will be a list of 'GroupMatchingResult' objects, the length of the list equals to number of different basis groups the algorithm returned.

To check all matched peaks for path 0 of group 3:
```python
#code example
final_matching_results[3].sum_matched_results_mona[0]['all_matched_peaks']

#output contains m/z values with corresponding intensities,  length of the output list equals to number of layers of recursive matching
[[(71.0158167500592, 47969.802360823625),
  (85.0317277974723, 6320.15038865462),
  (87.01097295864278, 6682.335842026297),
  (89.0266944921628, 248952.3833011955),
  (101.026740174181, 49280.89213672266),
  (113.026855612246, 70180.12405222801),
  (119.037313783605, 88633.76032844598),
  (131.03736927702698, 17388.240020534537),
  (143.037350098239, 34897.2601425634),
  (144.040767202612, 2446.428359306434),
  (149.047991556927, 13860.063281434182),
  (161.048064401475, 61930.59041665014),
  (163.052407743035, 903.4948256139173),
  (179.058959578171, 439580.0364151183),
  (90.0301153901944, 8713.775531721745),
  (102.029954287965, 2340.728504860962),
  (114.03005154321501, 4009.1582278610213),
  (120.040628388736, 4784.574694971612),
  (132.039424387647, 1360.207194000837),
  (162.051370452295, 4394.284055390238)],
 [(107.037093163663, 9467.584639432278)]]
```
To check candidates for first layer of path 0 of group 0:
```python
#code exmaple
final_matching_results[0].sum_matched_results_mona[0]['candi_list_each_matching_layer'][0]

#output is a list of candidates information(candidate compound,spectrum,similarity score)
[(MonaCompounds(id=1412, name='Creatine'),
  MonaSpectrum(mode='Negative', ms_level='MS2', n_peaks=2, reduced_n_peaks=2, name='Creatine', spectrum_id='PR100534'),
  0.9981105483381697),
 (MonaCompounds(id=2354, name='Creatine,anhydrous'),
  MonaSpectrum(mode='Negative', ms_level='MS2', n_peaks=2, reduced_n_peaks=2, name='Creatine,anhydrous', spectrum_id='PT201840'),
  0.9981105483381697),
 (MonaCompounds(id=7987, name='L-Alanine'),
  MonaSpectrum(mode='Negative', ms_level='MS2', n_peaks=2, reduced_n_peaks=2, name='L-Alanine', spectrum_id='RP000211'),
  0.9846983016063124),
 (MonaCompounds(id=1628, name='3-Guanidinopropionic acid'),
  MonaSpectrum(mode='Negative', ms_level='MS2', n_peaks=2, reduced_n_peaks=2, name='3-Guanidinopropionic acid', spectrum_id='PR100630'),
  0.9829381926557169),
 (MonaCompounds(id=8288, name='Indole-3-acetyl-L-alanine'),
  MonaSpectrum(mode='Negative', ms_level='MS2', n_peaks=4, reduced_n_peaks=4, name='Indole-3-acetyl-L-alanine', spectrum_id='RIKENPlaSMA008276'),
  0.9395831669260067),...]

```
## Demo
####
demo data could be found under /demo_data

Example usage of demo datafile:
```python
"""
database files can be retrieved through step 1
"""
mona_dire_neg = '../../../MoNA/mona_neg_cleaned.pkl'  # directory to store pickle file of MoNA database (negative mode)
mona_dire_pos = '../../../MoNA/mona_pos_cleaned.pkl'  # directory to store pickle file of MoNA database (positive mode)
mona = MonaDatabase(neg_dir=mona_dire_neg, pos_dir=mona_dire_pos)
mona.read_file(cur_mode='Negative')  # cur_mode: 'Negative' or 'Positive'
for c in tqdm(mona.compounds_list, desc="processing compounds"):
    c.generate_mz_collection(mode='Negative')  # mode: 'Negative' or 'Positive'

m = MSData.from_files('demo_data', '/.../.../demo_data')
m.apply_smooth_conv(m.gaussian_kernel(2, 12), rt_range=(0., np.inf))  # smooth the raw data of peaks
peak_detection_options = {'height': 0.1, 'range_threshold': 0.1, 'prominence': 0.1}
m.perform_feature_peak_detection(**peak_detection_options)  # detect peaks
m.remove_duplicates_detection()  # remove duplicates
by = np.max(m.ints_raw, axis=1)
m.sort_feature(by=by)  # sort features by intensity order(high to low)

####################################generate basis step####################################
gen_base_opt = dict(
    l1=2.,
    min_sin=5E-3,  
    min_base_err=5E-4, 
    min_rt_diff=1.,
    max_cos_sim=0.7
)
m.generate_base(reset=True, allowed_n_overlap=(1, 2), **gen_base_opt)
m.generate_base(reset=False, allowed_n_overlap=2, **gen_base_opt)
m.generate_base(reset=False, allowed_n_overlap=3, **gen_base_opt)
m.generate_base(reset=False, allowed_n_overlap=(4, None), **gen_base_opt)

####################################perform peak decompositon for other features####################################
m.perform_peak_decomposition(l1=1.)  # decompose other features using generated basis group
spec_params = dict(
    threshold=1E-5,
    max_rt_diff=4.,
    max_mse=np.inf,
    mz_upperbound=5E-2,
    min_cos=0.90)
for i in tqdm(range(m.n_base), desc='generating spectrum'):
    m.gen_spectrum(i, plot=False, load=False, **spec_params)  # generate reconstructed spectrum for each group

####################################database matching for each basis group####################################
final_matching_results = []  # create an empty list to contain the following matching result
start = datetime.datetime.now()
for i in range(len(m.base_index)):
    print(f'current running basis {i}')
    spec = m.base_info[m.base_index[i]].spectrum
    spec_2 = copy.deepcopy(spec)

    result = GroupMatchingResult(recons_spec=spec_2, base_index_relative=i, base_index_abs=m.base_index[i],
                                 mode='Negative')
    # mode: 'Negative' or 'Positive'

    result.gen_mona_matching_result(total_layer_matching=0, n_candidates_further_matched=3, database=mona,
                                    transform=None)
    # total_layer_matching: number of layers for recursive matching(any positive integer), 0 :no recursive mathing, 1: one further recursive mathching(recommended)
    # n_candidates_further_matched: number of top-candidates considered for further recursive matching
    # transform: None or math.sqrt

    result.summarize_matching_re_all_db()
    result.remove_db()
    final_matching_results.append(result)
end = datetime.datetime.now()
print(end - start)

```


