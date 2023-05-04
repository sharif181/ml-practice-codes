import numpy as np
import rapidfuzz.fuzz
from rapidfuzz import process
from rapidfuzz.distance.Levenshtein import normalized_similarity
from rapidfuzz import fuzz
import Levenshtein


def group_list_by_similarity(listN, score_cutoff=0.7, key=None, allow_duplicates=False):
    def convert_index_to_string(main_data_source, index_list, scores):
        # use set comprehension on index_list to avoid duplicate values.
        # index list is basically list of index which are in the same group
        # main_data_source is the main list of strings
        # this helper function will be used to convert index to string value
        # This function will return list of strings which are in same group, based on index_list

        # return list({(main_data_source[index], np.sum(scores[index])) for index in index_list})

        return [(main_data_source[index], np.sum(scores[index])) for index in index_list]

    if key is None:
        listN_orig = listN.copy()
        listM = listN
        scores = process.cdist(listN, listM, scorer=normalized_similarity, dtype=np.uint8,
                               score_cutoff=score_cutoff)

    else:
        listN_orig = listN
        listN = [x[key] for x in listN_orig]

        listM = listN
        scores = process.cdist(listN, listM, scorer=normalized_similarity, dtype=np.uint8,
                               score_cutoff=score_cutoff)

    # storing all col value in numpy array
    # Example: if len of main listN=5
    # then all_col=[0, 1, 2, 3, 4]
    all_col = np.arange(0, len(listN), dtype=int)

    # store all groups: [ [(String, value), (String, value)], [(String, value), (String, value)] ]
    result = []

    # This list will be using as queue.
    # We will add index of col that will need to be checked.
    # initially We will start from 0th col
    look_up_cols = []
    if len(all_col) > 0:
        look_up_cols = [0]

    while len(look_up_cols) > 0:
        # We will start from 0th col.
        # np.nonzero() will return indexes from 0th col where values are greater than 0
        # np.nonzero() will return tuple (list_of_indexes, dtype)
        # for list of value, accessing 0th index
        cols_values = np.nonzero(scores[:, look_up_cols[0]])[0]

        # this is just for little performance improvement
        rows_values = np.nonzero(scores[look_up_cols[0], :])[0]

        # storing list of same group.
        # cols_values will be numpy array, it will have indexes from columns, where values are greater than 0
        # convert_index_to_string will convert list_of_index to original strings from main list
        # Example: if cols_values=[0,1,2],
        # then convert_index_to_string will convert it [(original_string, score_value), (original_string, score_value), (original_string, score_value)]
        result.append(convert_index_to_string(listN_orig, cols_values, scores))

        # remove already visited col
        # as it is queue so removing first index
        del look_up_cols[0]

        # removing index from all_col
        # all_col initially have all the index from [0.....len(listN)]
        # removing index which are already in a group
        # use set for taking advantage, but we need list again so, convert it into list
        all_col = list(set(all_col) - set(cols_values))

        # adding indexes into look_up_cols
        # after checking a specific col, if any new index that are not in any group found,
        # then add those indexes in look_up_cols.
        # Use set union to add new indexes into look_up_cols,
        # if use simple list addtion then duplicate can be arised.
        look_up_cols = list(set(look_up_cols).union(set(all_col)))

        # this line of code will remove duplicate values from look_up_cols
        # It can be implemented in better way
        look_up_cols = list(set(look_up_cols) - set(rows_values))

    return result


def find_player_names_per_group(player_names, flag=False):
    player_names_cp = player_names.copy()
    string_groups = group_list_by_similarity(player_names)
    # string_group_values = [x for x in string_groups if len(x) > 2]
    string_group_values = string_groups

    string_group_values_median = []
    for item in string_group_values:
        string_group_values_median.append((Levenshtein.setmedian([x[0] for x in item]), item))

    names = [x[0] for x in string_group_values_median]

    string_group_values = []
    for name in names:
        store_name = process.extract(name, player_names_cp, scorer=fuzz.ratio, score_cutoff=75,
                                     limit=len(player_names_cp))
        if len(store_name) > 0:
            string_group_values.append(store_name)
            for n in store_name:
                if n[0] in player_names_cp:
                    player_names_cp.remove(n[0])

    string_group_values_median = []
    for item in string_group_values:
        string_group_values_median.append((Levenshtein.setmedian([x[0] for x in item]), item))

    if flag:
        return [x[0] for x in string_group_values_median]
    # return [x[0] for x in string_group_values_median if (len(x[1]) / len(player_names)) >= (1 / len(names))]
    return [x[0] for x in string_group_values_median if (len(x[1]) / len(player_names)) >= 0.1]


def create_group_indexes(result):
    if len(result) > 0:
        main_index = []
        # max_val = max(result)
        # min_val = min(result)
        min_val = result[0]
        max_val = result[-1]
        while min_val <= max_val:
            main_index.append(min_val)
            min_val += 1

        return main_index
    return result


def remove_outlayers(index_list, key):
    index_list = sorted(index_list)
    result = []
    for index, value in enumerate(index_list):
        if index == 0:
            result.append(value)
            continue

        if abs(value - index_list[index - 1]) <= 1:
            result.append(value)
        else:
            break

    return create_group_indexes(result)


def filter_row_values(index_values, key):
    result = []
    for index in index_values:
        if index >= key:
            result.append(index)
    result = sorted(result)
    return result


def convert_index_to_group(main_data_source, index_list, scores):
    temp_group = []
    # if len(index_list) == 1:
    #     temp_group.append(main_data_source[index_list[0][0]][0])
    #     return temp_group

    for index in index_list:
        temp_group.extend(main_data_source[index])

    # groups = []
    # temp_group_2 = []
    # for index in range(len(temp_group)):
    #     # group1 = temp_group[inx + 1]
    #     # group_1_stream_pos = group1['recording_data'].stream_position
    #     # group2 = temp_group[inx]
    #     # group_2_stream_pos = group2['recording_data'].stream_position
    #     # if group_1_stream_pos - group_2_stream_pos < 120000:
    #     temp_group_2.append(temp_group[index])
    #     # else:
    #     #     break
    # if len(temp_group_2) > 0:
    #     groups.append(temp_group_2)
    return temp_group


def group_merge_helper(listN, data_source, score_cutoff=70):
    merged_groups = []

    listM = listN
    string_match_scores = process.cdist(listN, listM, scorer=rapidfuzz.fuzz.ratio, dtype=np.uint8,
                                        score_cutoff=score_cutoff)

    # string_match_scores = np.array(group_matches_scores, dtype=int)
    all_row = np.arange(0, len(string_match_scores), dtype=int)
    look_up_rows = []
    if len(all_row) > 0:
        look_up_rows = [0]

    while len(look_up_rows) > 0:
        # give non zero values for one row
        row_values = np.nonzero(string_match_scores[look_up_rows[0], :])[0]
        # row_values = np.where(row_values >= look_up_rows[0])[0]
        row_values = filter_row_values(row_values, look_up_rows[0])
        row_values = remove_outlayers(row_values, look_up_rows[0])
        # row_values = self.reject_outliers(np.array(row_values))
        # row_values = self.reject_outliers(string_match_scores[look_up_rows[0], :])[0]
        # row_values = np.nonzero(row_values)[0]
        if len(row_values) == 0:
            del look_up_rows[0]
            del all_row[0]
            continue

        groups = convert_index_to_group(data_source, row_values, string_match_scores)
        merged_groups.append(groups)

        # if len(row_values) == 1:
        #     row_values = np.array(row_values[0])

        # give non zero values for one column
        # col_values = np.nonzero(string_match_scores[:, look_up_rows[0]])[0]
        # merged_groups.append(self.convert_index_to_group(athlete_groups, row_values, string_match_scores))
        # col_values = self.reject_outliers(string_match_scores[:, look_up_rows[0]])[0]
        # col_values = np.nonzero(col_values)[0]
        del look_up_rows[0]
        all_row = list(set(all_row) - set(row_values))
        look_up_rows = list(set(look_up_rows).union(set(all_row)))
        look_up_rows = list(set(look_up_rows) - set(row_values))
        look_up_rows = sorted(look_up_rows)
        # look_up_rows = list(set(look_up_rows) - set(col_values))
    return merged_groups
