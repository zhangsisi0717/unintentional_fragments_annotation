import pandas as pd
from pandas import DataFrame
# from type_checking import *
# file = '/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result/neg_mse5E3_rangethre0.05_Ker2_12_mincos85/iroa_name_final.csv'
##.csv files with ground truth, including


def check_base(index, m, file):
    iroa_df = pd.read_csv(file)
    temp_list = [j for j in m.base_info[m.base_index[index]].spectrum.spectrum_list if j[1] > 0.00]
    temp_list.sort(key=lambda x: x[1], reverse=True)
    cmp_info = []
    for mz, ints in temp_list:
        new_df = iroa_df[(abs(iroa_df['M-H'] - mz) / mz) * 1E6 < 30]
        rt_match_df = new_df[abs(new_df['RT(S)'] - m.base_info[m.base_index[index]].rt) <= 6]
        names = rt_match_df['Name'].values
        mz = rt_match_df['M-H'].values
        retenT = rt_match_df['RT(S)'].values
        inchikey = rt_match_df['InChiKey'].values
        re = list(zip(names, mz, retenT, inchikey))
        cmp_info.append(re)
    new_cmp_info = [j for i in cmp_info if i for j in i]
    return m.base_info[m.base_index[index]].rt, new_cmp_info, temp_list


def correct_cmp_iroa(base_rela_index, final_matching_results, m, file):
    rt, cmp_info, temp_list = check_base(base_rela_index, m, file)
    matched_cmp = []
    matched_cmp_name = []
    for cmp in cmp_info:
        if len(matched_cmp) < len(cmp_info):
            if cmp[0] not in matched_cmp_name:
                for idx, content in final_matching_results[base_rela_index].sum_matched_results_iroa.items():
                    if cmp[0] not in matched_cmp_name:
                        for layer in content['candi_list_each_matching_layer']:
                            if cmp[0] not in matched_cmp_name:
                                for index in range(len(layer)):
                                    if cmp[3] == layer[index][0].InChIKey:
                                        matched_cmp.append((base_rela_index, cmp[0], layer[index][2], index,
                                                            cmp[1], cmp[2], cmp[3]))
                                        matched_cmp_name.append(cmp[0])
                                        break

    return matched_cmp


def correct_cmp_mona(base_rela_index, final_matching_results, m, file):
    rt, cmp_info, temp_list = check_base(base_rela_index, m, file)
    matched_cmp = []
    matched_cmp_name = []
    for cmp in cmp_info:
        if len(matched_cmp) < len(cmp_info):
            if cmp[0] not in matched_cmp_name:
                for idx, content in final_matching_results[base_rela_index].sum_matched_results_mona.items():
                    if cmp[0] not in matched_cmp_name:
                        for layer in content['candi_list_each_matching_layer']:
                            if cmp[0] not in matched_cmp_name:
                                for index in range(len(layer)):
                                    if cmp[3] == layer[index][0].InChIKey:
                                        matched_cmp.append((base_rela_index, cmp[0], layer[index][2], index,
                                                            cmp[1], cmp[2], cmp[3]))
                                        matched_cmp_name.append(cmp[0])
                                        break

    return matched_cmp


def correct_cmp_mzc(base_rela_index, final_matching_results, m, file):
    rt, cmp_info, temp_list = check_base(base_rela_index, m, file)
    matched_cmp = []
    matched_cmp_name = []
    for cmp in cmp_info:
        if len(matched_cmp) < len(cmp_info):
            if cmp[0] not in matched_cmp_name:
                for idx, content in final_matching_results[base_rela_index].sum_matched_results_mzc.items():
                    if cmp[0] not in matched_cmp_name:
                        for layer in content['candi_list_each_matching_layer']:
                            if cmp[0] not in matched_cmp_name:
                                for index in range(len(layer)):
                                    if cmp[3] == layer[index][0].InChIKey:
                                        matched_cmp.append((base_rela_index, cmp[0], layer[index][2], index,
                                                            cmp[1], cmp[2], cmp[3]))
                                        matched_cmp_name.append(cmp[0])
                                        break

    return matched_cmp

# def correct_cmp_mzc(base_rela_index, final_matching_results, m, file):
#     rt, cmp_info, temp_list = check_base(base_rela_index, m, file)
#     matched_cmp = []
#     matched_cmp_name = []
#     skip_this_layer = False
#     for idx, content in final_matching_results[base_rela_index].sum_matched_results_mzc.items():
#         if len(matched_cmp) < len(cmp_info):
#             for cmp in cmp_info:
#                 if cmp[0] not in matched_cmp_name:
#                     for first_layer in content['candi_list_each_matching_layer']:
#                         if not skip_this_layer:
#                             if first_layer:
#                                 for index in range(len(first_layer)):
#                                     if cmp[3] == first_layer[index][0].InChIKey:
#                                         matched_cmp.append((base_rela_index, cmp[0], first_layer[index][2], index,
#                                                             cmp[1], cmp[2], cmp[3]))
#                                         matched_cmp_name.append(cmp[0])
#                                         skip_this_layer = True
#                                         break
#                         else:
#                             skip_this_layer = False
#                             break
#         else:
#             break
#     return matched_cmp


def gen_df_for_correct_annotation(database, final_matching_results, m, file):
    if database == 'iroa':
        iroa_final_result = []
        iroa_mis_matched = []
        for i in range(len(final_matching_results)):
            iroa_temp = correct_cmp_iroa(i, final_matching_results, m, file)
            if iroa_temp:
                iroa_final_result += iroa_temp
            else:
                iroa_mis_matched.append(i)
        iroa_df_final = DataFrame(iroa_final_result,
                                  columns=['base_idx', 'Name', 'match_score', 'nth_candidate', 'mz', 'rt', 'InChIKey'])
        return iroa_df_final, iroa_mis_matched

    if database == 'mona':
        mona_final_result = []
        mona_mis_matched = []
        for i in range(len(final_matching_results)):
            mona_temp = correct_cmp_mona(i, final_matching_results, m, file)
            if mona_temp:
                mona_final_result += mona_temp
            else:
                mona_mis_matched.append(i)
        mona_df_final = DataFrame(mona_final_result,
                                  columns=['base_idx', 'Name', 'match_score', 'nth_candidate', 'mz', 'rt', 'InChIKey'])
        return mona_df_final, mona_mis_matched

    if database == 'mzc':
        mzc_final_result = []
        mzc_mis_matched = []
        for i in range(len(final_matching_results)):
            mzc_temp = correct_cmp_mzc(i, final_matching_results, m, file)
            if mzc_temp:
                mzc_final_result += mzc_temp
            else:
                mzc_mis_matched.append(i)
        mzc_df_final = DataFrame(mzc_final_result,
                                 columns=['base_idx', 'Name', 'match_score', 'nth_candidate', 'mz', 'rt', 'InChIKey'])
        return mzc_df_final, mzc_mis_matched


def gen_file_matched_bases(m, file):####generate correct peak info within each basis based on input rt,mz
    total_re = []
    for base_idx in range(len(m.base_index)):
        temp_result = check_base(base_idx, m, file)[1]
        for i in range(len(temp_result)):
            temp_result[i] = list(temp_result[i])
            temp_result[i] = [base_idx] + temp_result[i]
        total_re += temp_result

    df_final = DataFrame(total_re,columns=['base_idx', 'Name','mz', 'rt', 'InChIKey'])
    return df_final

