import json
from tqdm.auto import tqdm
import pickle as pkl

"""
User could download MoNA database (experimental LC-MSMS data) at https://mona.fiehnlab.ucdavis.edu/downloads

"""

with open('MoNA-export-LC-MS-MS_Negative_Mode.json') as f:  # open the json file for either positive or negative mode
    mona_raw = json.load(f)

mona_spectra = dict()
for i in tqdm(range(len(mona_raw))):
    temp = []
    spec_id = mona_raw[i].get('id')
    inchi = mona_raw[i]['compound'][0].get('inchi')
    inchiKey = mona_raw[i]['compound'][0].get('inchiKey')
    spec = mona_raw[i].get('spectrum')
    if not spec: continue
    mz = []
    intensity = []
    index = 0
    for jd in range(index + 1, len(spec)):  ##parse spectrum
        if spec[jd] == ':':
            mz.append(float(spec[index:jd]))
            index = jd + 1

        elif spec[jd] == ' ':
            intensity.append(float(spec[index:jd]))
            index = jd + 1

        elif jd == len(spec) - 1:
            intensity.append(float(spec[index:]))
    spectrum_list = list(zip(mz, intensity))
    if len(spectrum_list) >= 100:  # could be any number of peaks
        continue

    """
    Further filtering the spectra data by excluding spectra containing more than certain number of peaks is important.
    Without this filtering step, it would take a very long time in further database matching step.
    
    """

    temp += [('spectrum_id', spec_id), ('spectrum_list', spectrum_list), ('inchi', inchi), ('inchiKey', inchiKey)]

    for j in mona_raw[i]['compound'][0]['metaData']:  ##cmp_info##
        if j['name'] == 'molecular formula':
            formula = j['value']
            temp.append(('molecular formula', formula))
        if j['name'] == 'total exact mass':
            exact_mass = j['value']
            temp.append(('total exact mass', exact_mass))
    for k in mona_raw[i]['metaData']:  ##meta_data
        if k['name'] == 'ms level':
            ms_level = k['value']
            temp.append(('ms level', ms_level))
        elif k['name'] == 'collision energy':
            collision_energy = k['value']
            temp.append(('collision energy', collision_energy))
        elif k['name'] == 'precursor m/z':
            precursor = k['value']
            temp.append(('precursor m/z', precursor))
        elif k['name'] == 'precursor type':
            pre_type = k['value']
            temp.append(('precursor type', pre_type))
    mona_spectra[i] = dict(temp)

with open('file_name.pkl', 'wb') as f:
    pkl.dump(mona_spectra, f)
