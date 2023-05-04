import numpy as np
from rapidfuzz import process, fuzz
from rapidfuzz.distance.Levenshtein import normalized_similarity

import Levenshtein


def find_most_common_strings_and_scores(athlete_groups):
    modified_groups = []
    for inx, athlete_group in enumerate(athlete_groups):
        # athlete_names = []
        athlete_names = [a['athlete_name'].upper() for athletes in [x['athletes'] for x in athlete_group] for a in
                         athletes]
        # for athletes in [x['athletes'] for x in athlete_group]:
        #     athlete_names.extend([x['athlete_name'].upper() for x in athletes])

        group_athlete_names = find_player_names_per_group(athlete_names)
        if len(group_athlete_names) > 1:
            athlete_conf = []
            for item in group_athlete_names:
                score = 0
                player_count = len(process.extract(item, athlete_names, scorer=fuzz.ratio, score_cutoff=75,
                                                   limit=len(athlete_names)))
                if player_count > 0:
                    score = round((player_count / len(athlete_names)) * 100, 2)
                athlete_conf.append((item, score))

            modified_groups.append((athlete_group, athlete_conf, group_athlete_names))
    return modified_groups


def find_player_names_per_group(player_names):
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
    return [x[0] for x in string_group_values_median]


def group_list_by_similarity(listN, data_source=None, score_cutoff=0.7, key=None, allow_duplicates=False):
    def convert_index_to_string(main_data_source, index_list, scores):
        # use set comprehension on index_list to avoid duplicate values.
        # index list is basically list of index which are in the same group
        # main_data_source is the main list of strings
        # this helper function will be used to convert index to string value
        # This function will return list of strings which are in same group, based on index_list

        # return list({(main_data_source[index], np.sum(scores[index])) for index in index_list})
        res = []
        for index in index_list:
            res.extend(main_data_source[index])
        # return [(main_data_source[index], np.sum(scores[index])) for index in index_list]
        return res

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
        result.append(convert_index_to_string(data_source, cols_values, scores))

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


def find_similarity_between_groups(groups):
    group_match_scores = []
    for i in range(len(groups)):
        scores_list = []
        first_athlete_names = groups[i]

        for j in range(len(groups)):
            second_athlete_names = groups[j]
            count = 0
            if len(first_athlete_names) < len(second_athlete_names):
                for item in first_athlete_names:
                    if process.extractOne(item, second_athlete_names, scorer=normalized_similarity,
                                          score_cutoff=0.6):
                        count += 1
                score = round(count / len(second_athlete_names), 2) * 100
            else:
                for item in second_athlete_names:
                    if process.extractOne(item, first_athlete_names, scorer=normalized_similarity,
                                          score_cutoff=0.6):
                        count += 1
                score = round(count / max(len(first_athlete_names), count, 1), 2) * 100
            if score >= 25.0:
                scores_list.append(score)
            else:
                scores_list.append(0)
        group_match_scores.append(scores_list)
    return np.array(group_match_scores)
