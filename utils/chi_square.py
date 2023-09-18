#!/usr/bin/env python3

import scipy
import sys

def chi_square_likelihood(observed, expected):
    # observed and expected are dicts: id -> cnt

    # First, let's map each id to an integer
    id2i = {}
    i=0
    for k in expected:
        if not k in id2i:
            id2i[k] = i
            i += 1
    for k in observed:
        if not k in id2i:
            id2i[k] = i
            i += 1
    n = i # the length of the array
    observed_arr = [0] * n
    expected_arr = [0] * n
    for k in observed:
        observed_arr[id2i[k]] += observed[k]
    for k in expected:
        expected_arr[id2i[k]] += expected[k]

    print(observed_arr)
    print(expected_arr)

    chisq, p = scipy.stats.chisquare(observed_arr, expected_arr)
    return p

if __name__ == "__main__":
    p = chi_square_likelihood({123:1,124:1,125:2},{123:1,124:1,125:1,126:1})
    print("uniformity conficende ",p*100,"%")

    # observed = [5,8,9,8,10,20]
    observed = [10,10,10,10,5,15]
    expected = [10, 10, 10, 10, 10, 10]
    chisq, p = scipy.stats.chisquare(observed, expected)
    print("experimental sum=",chisq)
    print("p value=",p)
    print("uniformity conficende ",p*100,"%")
    print("uniformity unlikedness ",(1-p)*100,"%")

