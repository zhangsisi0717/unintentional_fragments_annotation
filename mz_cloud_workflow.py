from mzcloud import *
def read_mzc_data(mode):
    if mode not in ('Positive','Negative'):
        raise ValueError('mode has to be Positive or Negative!')
    # mz_cloud_dir = "../../mz_cloud"
    mz_cloud_dir = '/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/mz_cloud'
    mzc = MZCloud(mz_cloud_dir)
    mzc.read_comp_metadata()
    # def filter_func(x: MZCloudSpectrum):
    #     # return x.SpectrumKind == 'Normal spectrum'
    #     return x.SpectrumKind == 'Average spectrum'
    for c in tqdm(mzc.compounds, desc="reading MZcloud compounds"):
        c.generate_mz_collection(threshold=1E-3, mode=mode) #Negative or Positive#
        c.get_precursor(mode=mode)  #Negative or Positive#



