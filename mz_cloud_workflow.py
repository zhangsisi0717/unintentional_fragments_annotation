from mzcloud import *
# mz_cloud_dir = "../../mz_cloud"
mz_cloud_dir= '/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/mz_cloud'
mzc = MZCloud(mz_cloud_dir)
mzc.read_comp_metadata()


def filter_func(x: MZCloudSpectrum):
    # return x.SpectrumKind == 'Normal spectrum'
    return x.SpectrumKind == 'Average spectrum'


for c in tqdm(mzc.compounds, desc="processing compounds"):
    c.generate_mz_collection(threshold=1E-3, mode='Negative')
    c.get_precursor(mode='Negative')

lucose = []
for c in tqdm(mzc.compounds):
    if 'lucose' in c.CompoundName:
        lucose.append(c)

glc = lucose[6].spectra_2[1][1]
re = iroa.find_match(target=glc,save_matched_mz=True,transform=math.sqrt)
