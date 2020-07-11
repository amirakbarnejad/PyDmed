

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
