from check_annotation_accurary import *
import copy

def gen_result_groundtruth(m,goundtruth_file_path,path_to_store_xlx,path_to_store_csv):
    df_re_truth=gen_file_matched_bases(m=m, file=goundtruth_file_path)
    df_re_truth.to_excel(path_to_store_xlx, index=False)
    df_re_truth.to_csv(path_to_store_csv,index=False)

def gen_correct_matched_results(m,database='mzc',final_matching_results=None,goundtruth_file_path=None,path_store_xlx=None,path_store_csv=None):
    re_df,mismatch_idx = gen_df_for_correct_annotation(database, final_matching_results, m, goundtruth_file_path)
    re_df.to_excel(path_store_xlx, index=False)
    re_df.to_csv(path_store_csv,index=False)

def gen_not_matched_results(database,df_truth_path=None,correct_match_path=None,path_to_store=None):
    df_truth_0 = pd.read_csv(df_truth_path)
    df_db = pd.read_csv(correct_match_path)
    db_have_key=[]
    db_not_have_key=[]
    if database.name == 'mzc':
        for idx in range(len(df_truth_0['InChIKey'].values)):
            not_found = True
            for cmp in database.compounds:
                if cmp.InChIKey == df_truth_0['InChIKey'].values[idx]:
                    db_have_key.append(idx)
                    not_found = False
                    break
            if not_found:
                db_not_have_key.append(idx)

    elif database.name == 'iroa' or database.name == 'mona':
        for idx in range(len(df_truth_0['InChIKey'].values)):
            not_found = True
            for cmp in database.compounds_list:
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
            for cmp in database.compounds:
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

    elif database.name == 'iroa':
        for index in range(len(not_matched_df['InChIKey'].values)):
            target_cmp=[]
            for cmp in database.compounds_list:
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
    elif database.name == 'mona':
        for index in range(len(not_matched_df['InChIKey'].values)):
            target_cmp=[]
            for cmp in database.compounds_list:
                if cmp.InChIKey == not_matched_df['InChIKey'].values[index]:
                    target_cmp.append(cmp)
                    break
            if target_cmp:
                not_matched_df.iloc[[index],[5]] = 'Y'
                polarity=set()
                for s in target_cmp[0].spectra_1 + target_cmp[0].spectra_2:
                    polarity.add(s.mode)
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


def gen_all_db_results(m,mona_path,iroa_path,mzc_path,save_path_xls,save_path_csv,iroa,mzc,mona):
    df_iroa = pd.read_csv(iroa_path)
    df_mona = pd.read_csv(mona_path)
    df_mzc = pd.read_csv(mzc_path)
    df_iroa['database']='iroa'
    df_mona['database']='mona'
    df_mzc['database']='mzc'

    matched_all_db=pd.concat([df_iroa, df_mona, df_mzc])
    matched_all_db.drop_duplicates()
    new_cols = ['spec_matched_peaks','non_matched','num_matched','isotope','num_isotope','adduction','num_adduction','multimer','num_multimer']
    for i in new_cols:
        matched_all_db[i] = 'NA'
    matched_all_db['spec_matched_peaks'] = matched_all_db['spec_matched_peaks'].apply(lambda x: list())
    matched_all_db['non_matched'] = matched_all_db['non_matched'].apply(lambda x: list())
    matched_all_db['isotope'] = matched_all_db['isotope'].apply(lambda x: dict())
    matched_all_db['adduction'] = matched_all_db['adduction'].apply(lambda x: dict())
    matched_all_db['multimer'] = matched_all_db['multimer'].apply(lambda x: dict())
    matched_all_db.reset_index(drop=True,inplace=True)

    for idx in tqdm(range(len(matched_all_db.index))):
        base_idx = matched_all_db.loc[idx, 'base_idx']
        temp_spec = copy.deepcopy(m.base_info[m.base_index[base_idx]].spectrum)
        inchikey = matched_all_db.loc[idx, 'InChIKey']
        if matched_all_db.loc[idx, 'database'] == 'iroa':
            mat, non_mat,iso,add,mul = temp_spec.gen_matched_peaks_compound(inchIkey=inchikey, mode='Positive', database=iroa)
        elif matched_all_db.loc[idx, 'database'] == 'mona':
            mat, non_mat,iso,add,mul = temp_spec.gen_matched_peaks_compound(inchIkey=inchikey, mode='Positive', database=mona)
        elif matched_all_db.loc[idx, 'database'] == 'mzc':
            mat, non_mat,iso,add,mul = temp_spec.gen_matched_peaks_compound(inchIkey=inchikey, mode='Positive', database=mzc)

        matched_all_db.at[idx, 'spec_matched_peaks'] = mat
        matched_all_db.at[idx, 'non_matched'] = non_mat
        matched_all_db.at[idx, 'isotope'],matched_all_db.at[idx, 'adduction'], matched_all_db.at[idx, 'multimer']= iso,add,mul

        matched_all_db.iloc[idx,10] = len(mat)
        matched_all_db.iloc[idx,12],matched_all_db.iloc[idx,14], matched_all_db.iloc[idx,16] = len(iso),len(add),len(mul)

    matched_all_db.to_excel(save_path_xls,index=False)
    matched_all_db.to_csv(save_path_csv,index=False)







