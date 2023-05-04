import numpy as np
from rapidfuzz import process, fuzz
from rapidfuzz.distance.Levenshtein import normalized_similarity

import Levenshtein
from rapidfuzz.process import extractOne
from rapidfuzz.fuzz import ratio

#
# s1 = 'A. LIMBACHER H. SCHMIDT J. GRILLET K. OFNER'
# s2 = 'H. SCHMIDT J. GRILLET'
#
# value = ['B. MIDOL R. REGEZ V. SHMYROV Y. GUNSCH', 'B. MIDOL R. REGEZ Y. GUNSCH',
#          'D. MOBAERG K. SYSOEV S. DEROMEDIS S. SIEBENHOFER', 'S. DEROMEDIS S. SIEBENHOFER',
#          'F. WILMSMANN J. BERRY T. HRONEK V. ANDERSSON', 'B. LEMAN J. ROHRWECK M. BISCHOFBERGER S. FURUNO',
#          'B. LEMAN J. ROHRWECK S. FURUNO', 'A. KAPPACHER G. ROWELL K. DRURY L. TCHIKNAVORIAN T. TCHIKNAVORIAN',
#          'E. ZORZI I. OMELIN J. SCHMIDT L. OMELIN N. LEHIKOINEN', 'E. ZORZI N. LEHIKOINEN',
#          'C. COOK J. AUJESKY R. DETRAZ T. TAKATS', 'A. FIVA J. CHAPUIS L. LO N. BACHSLEITNER O. DAVIES', 'K. OLIVER',
#          'J. CHAPUIS', 'B. PHELAN S. NAESLUND T. GAIRNS T. GANTENBEIN', 'A. EDEBO C. HOFFOS E. MALTSEVA F. SMITH',
#          'I. SHERRET L. SHERRET M. THOMPSON N. SHERINA Z. CHORE', 'A. LIMBACHER H. SCHMIDT J. GRILLET K. OFNER',
#          'H. SCHMIDT J. GRILLET', 'A. LIMBACHER', 'B. MIDOL D. MOBAERG R. REGEZ S. DEROMEDIS',
#          'B. LEMAN J. ROHRWECK T. HRONEK V. ANDERSSON', 'N. BACHSLEITNER T. TAKATS',
#          'A. KAPPACHER E. ZORZI J. SCHMIDT T. TCHIKNAVOALAN T. TCHIKNAVORIAN', 'A. KAPPACHER E. ZORZI J. SCHMIDT',
#          'C. COOK N. BACHSLEITNER O. DAVIES T. TAKATS', 'C. HOFFOS F. SMITH S. NAESLUND T. GANTENBEIN',
#          'H. SCHMIDT J. GRILLET M. THOMPSON Z. CHORE', 'B. LEMAN D. MOBAERG J. ROHRWECK S. DEROMEDIS',
#          'C. COOK N. BACHSLEITNER T. TCHIKNAVORIAN', 'C. COOK J. SCHMIDT N. BACHSLEITNER T. TCHIKNAVORIAN',
#          'C. HOFFOS H. SCHMIDT J. GRILLET T. GANTENBEIN', 'H. SCHMIDT T. GANTENBEIN',
#          'F. SMITH M. THOMPSON S. NAESLUND Z. CHORE', 'B. LEMAN C. COOK C. COOOK N. BACHSLEITNER S. DEROMEDIS',
#          'D. MOBAERG J. ROHRWECK J. SCHMIDT T. TCHIKNAVORIAN']
#
# # score = process.cdist(value, value, score_cutoff=0.25, scorer=normalized_similarity)
# score = process.cdist(value, value, scorer=normalized_similarity, dtype=np.uint8,
#                        score_cutoff=0.4)
# print(score)
# s1 = ['E. BICHON L. BOZZOLO O. VISINTIN']
# s2 = ['E. BICHON G. BLOIS']
# s1 = ['G. HERPIN K. VUAGNOUX Y. TAKAHARA']
# s2 = ['M. SURGET']
s1 = ['E. GRONDIN', 'L. LE', 'Q. SODOGAS']
# s1 = ['E. Kuddus', 'P. Pum', 'C. DULL']
s2 = ['L. IE', 'L. SOMMARIVA', 'Q. SODOGAS', 'S. Khan']
# score = process.cdist(s1, s2, scorer=fuzz.ratio)
# print(score)
# s1 = 'GAEL ZULAUF'
# s2 = 'GILLES ROULIN'
# athlete_extracted = extractOne(s1, [s2],
#                               scorer=ratio
#                               )
# print(athlete_extracted)


score = process.cdist(s1, s2, scorer=ratio, dtype=np.uint8, score_cutoff=80)
print()