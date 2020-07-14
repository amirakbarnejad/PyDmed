

import numpy as np
import math



def lcm(list_numbers):
    '''
    Computes the lcm of numbers in a list.
    '''
    lcm = list_numbers[0]
    for idx_number in range(1, len(list_numbers)):
        lcm = (lcm*list_numbers[idx_number])/math.gcd(lcm, list_numbers[idx_number])
        lcm = int(lcm)
    return lcm
    
    
def multimode(list_input):
    '''
    `statistics.multimode` does not exist in all python versions.
    Therefore, we minimath.multimode is implemented.
    '''
    set_data = set(list_input)
    dict_freqs = {val:0 for val in set_data}
    for elem in list_input:
        dict_freqs[elem] = dict_freqs[elem] + 1
    mode = max((v, k) for k, v in dict_freqs.items())[1]
    return mode
        
