# Copyright (c) Tumiz.
# Distributed under the terms of the GPL-3.0 License.

def over(v,low,high):
    if v>high:
        return v-high
    elif v<=high and v>=low:
        return 0
    else:
        return v-low
    
def sign(x):
    if x >= 0:
        return 1
    else:
        return -1