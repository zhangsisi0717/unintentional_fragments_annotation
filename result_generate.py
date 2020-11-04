# goundtruth_file_path = '/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result_pos/iroa_name_final_pos.csv'
# path_to_store='/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result_pos/pos_ms1_matching_1'
# df_re_truth=gen_file_matched_bases(m=m, file=goundtruth_file_path)
# df_re_truth.to_excel(path_to_store+'/'+'df_truth_0.00.xlsx', index=False)
# df_re_truth.to_csv(path_to_store+'/'+'df_truth_0.00.csv',index=False)
#
#
# re_df,mismatch_idx = gen_df_for_correct_annotation('mzc', final_matching_results, m, goundtruth_file_path)
# re_df.to_excel(path_to_store+'/'+'mzc_correct_match_0.00.xlsx', index=False)
# re_df.to_csv(path_to_store+'/'+'mzc_correct_match_0.00.csv',index=False)
from check_annotation_accurary import *

def gen_result_groundtruth(m,goundtruth_file_path,path_to_store_xlx,path_to_store_csv):
    df_re_truth=gen_file_matched_bases(m=m, file=goundtruth_file_path)
    df_re_truth.to_excel(path_to_store_xlx, index=False)
    df_re_truth.to_csv(path_to_store_csv,index=False)

# path_to_store+'/'+database+'_correct_match_0.00.xlsx'
# path_to_store+'/'+database+'_correct_match_0.00.xlsx'
def gen_correct_matched_results(m,database='mzc',final_matching_results=None,goundtruth_file_path=None,path_store_xlx=None,path_store_csv=None):
    re_df,mismatch_idx = gen_df_for_correct_annotation(database, final_matching_results, m, goundtruth_file_path)
    re_df.to_excel(path_store_xlx, index=False)
    re_df.to_csv(path_store_csv,index=False)

# cur_path = '/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result_pos/pos_ms1_matching_1'
# df_truth_path = cur_path + '/' + 'df_truth_0.00.csv'
# correct_match_path = cur_path + '/' + 'mzc_correct_match_0.00.csv'
# path_to_store = cur_path+'/'+'not_matched_mzc_new2.xlsx'

def gen_not_matched_results(database,df_truth_path=None,correct_match_path=None,path_to_store=None):
    df_truth_0 = pd.read_csv(df_truth_path)
    df_db = pd.read_csv(correct_match_path)
    db_have_key=[]
    db_not_have_key=[]
    if database.name == 'mzc':
        for idx in range(len(df_truth_0['InChIKey'].values)):
            not_found = True
            for cmp in database.compounds:  ########################need to modify ##
                if cmp.InChIKey == df_truth_0['InChIKey'].values[idx]:
                    db_have_key.append(idx)
                    not_found = False
                    break
            if not_found:
                db_not_have_key.append(idx)

    elif database.name == 'iroa' or database.name == 'mona':
        for idx in range(len(df_truth_0['InChIKey'].values)):
            not_found = True
            for cmp in database.compounds_list:  ########################need to modify ##
                if cmp.InChIKey == df_truth_0['InChIKey'].values[idx]:
                    db_have_key.append(idx)
                    not_found = False
                    break
            if not_found:
                db_not_have_key.append(idx)

    db_not_matched=[]
    for idx in db_have_key:
        if df_truth_0.iloc[idx].InChIKey not in df_db['InChIKey'].values:
            db_not_matched.append(idx)

    not_matched_df = df_truth_0.reindex(db_not_matched)
    not_matched_df['have_inchi']='NA'
    not_matched_df['Neg_spec']='NA'
    not_matched_df['Pos_spec']='NA'

    if database.name == 'mzc':
        for index in range(len(not_matched_df['InChIKey'].values)):
            target_cmp=[]
            for cmp in database.compounds:  ##############change here###########
                if cmp.InChIKey == not_matched_df['InChIKey'].values[index]:
                    target_cmp.append(cmp)
                    break
            if target_cmp:
                not_matched_df.iloc[[index],[5]] = 'Y'
                polarity=set()
                for t in target_cmp[0].spectra_1 + target_cmp[0].spectra_2:
                   for s in t:
                        polarity.add(s.Polarity)
                if len(polarity)==1:
                    if 'Negative' in polarity:
                        not_matched_df.iloc[[index],[6]] = 'Y'
                        not_matched_df.iloc[[index],[7]] = 'N'
                    elif 'Positive' in polarity:
                        not_matched_df.iloc[[index],[6]] = 'N'
                        not_matched_df.iloc[[index],[7]] = 'Y'

                elif len(polarity) == 2:
                    not_matched_df.iloc[[index],[6]] = 'Y'
                    not_matched_df.iloc[[index],[7]] = 'Y'
            else:
                not_matched_df.iloc[[index],[5]] = 'N'

    elif database.name == 'iroa' or database.name == 'mona':
        for index in range(len(not_matched_df['InChIKey'].values)):
            target_cmp=[]
            for cmp in database.compounds_list:  ##############change here###########
                if cmp.InChIKey == not_matched_df['InChIKey'].values[index]:
                    target_cmp.append(cmp)
                    break
            if target_cmp:
                not_matched_df.iloc[[index],[5]] = 'Y'
                polarity=set()
                for s in target_cmp[0].spectra_1 + target_cmp[0].spectra_2:
                    polarity.add(s.Polarity)
                if len(polarity)==1:
                    if 'Negative' in polarity:
                        not_matched_df.iloc[[index],[6]] = 'Y'
                        not_matched_df.iloc[[index],[7]] = 'N'
                    elif 'Positive' in polarity:
                        not_matched_df.iloc[[index],[6]] = 'N'
                        not_matched_df.iloc[[index],[7]] = 'Y'

                elif len(polarity) == 2:
                    not_matched_df.iloc[[index],[6]] = 'Y'
                    not_matched_df.iloc[[index],[7]] = 'Y'
            else:
                not_matched_df.iloc[[index],[5]] = 'N'

    not_matched_df.to_excel(path_to_store, index=False)








