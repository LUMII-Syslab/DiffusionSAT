#!/usr/bin/env python3

from unqlite import UnQLite
import sys
import random
import matplotlib.pyplot as plt
#import utils.chi_square as chi2
from scipy import stats
import numpy as np

def fetch_item_by_attribute(table, attr_name, attr_value):
    for item in table.all():
        print(item, attr_value, attr_name)
        if item.get(attr_name) == attr_value:
            return item
    return None

def binary_cross_entropy(y_true, y_pred):
    sum = np.sum(y_true)
    y_true = np.true_divide(y_true, sum)
    
    sum = np.sum(y_pred)
    y_pred = np.true_divide(y_pred, sum)
    
    #print("Y_TRUE",y_true)
    #print("Y_PRED",y_pred)
    
    epsilon = 1e-15  # Small value to prevent log(0) errors
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predicted values to avoid log(0)
    
    # Calculate cross-entropy
    ce = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # Return the mean cross-entropy
    return np.mean(ce)


def inverse_arr(a):
    map = {}
    x = []
    y = []
    for pair in a:
        cnt = pair[1] # pair[0] -> pair[1] === sample -> cnt
        if cnt in map:
            map[cnt]+=1
        else:
            map[cnt]=1
    #print("MAP", map)

    for cnt, n_samples in dict(sorted(map.items())).items():
        x.append(cnt)
        y.append(n_samples)
    return x,y


def entropy_arr(a, middle, delta):
    map = {}
    x = []
    y = []
    for pair in a:
        cnt = pair[1] # pair[0] -> pair[1] === sample -> cnt
        
        if cnt in map:
            map[cnt]+=1
        else:
            map[cnt]=1
    
    for i in range(middle-delta-1,middle+delta+1):
        if map.get(i) is None:
            map[i] = 0
        
    before = 0
    after = 0
    for cnt, n_samples in dict(sorted(map.items())).items():
        #print(cnt," ",n_samples)
        if cnt<middle-delta:
            before += n_samples
            continue
        if cnt>middle+delta:
            after += n_samples
            continue
        x.append(cnt)
        y.append(n_samples)
        
    x = [middle-delta-1] + x + [middle+delta+1]
    y = [before] + y + [after]
    print("ENTRYOPY ARRS")
    print("x",x)
    print("y",y)
    return x,y

if __name__ == '__main__':
    db = UnQLite(filename="benchmarks.unqlite")
    table = db.collection('benchmarks')       


    n_iter = 3
    for item in table.all():
        if n_iter == 0:
            break
        n_iter -= 1
        
        n_uniform_samples = item["n_solutions"]*10
        #n_uniform_samples = sum(item["unigen_map"])
            
        uniform_cnt = [0]*item["n_solutions"]
        uniform_map = []

        ideal_cnt = [10]*item["n_solutions"]

        for i in range(n_uniform_samples):
            random_element =random.choice(range(item["n_solutions"]))                
            uniform_cnt[random_element] += 1
        for i in range(len(uniform_cnt)):
            uniform_map.append([i, uniform_cnt[i]])

        
        unigen_cnt = [row[1] for row in item["unigen_map"]]
        diffusion_cnt = [row[1] for row in item["diffusion_3-sat-unigen-500k_map"]]
        quicksampler_cnt = [row[1] for row in item["quicksampler_map"]]
        print("UNIGEN",len(unigen_cnt),unigen_cnt)
        print("DIFFUSION", len(diffusion_cnt), diffusion_cnt)
        print("UNIFORM", len(uniform_cnt), uniform_cnt)
        print("QUICKSAMPLER", len(quicksampler_cnt), quicksampler_cnt)
        print("IDEAL", len(ideal_cnt), ideal_cnt)
        print("")
        chisq, p  = stats.chisquare(unigen_cnt, ideal_cnt)
        print("UNIGEN/IDEAL chi2 uniformity probability=",p*100,"%")
        chisq, p  = stats.chisquare(diffusion_cnt, ideal_cnt)
        print("DIFFUSION/IDEAL chi2 uniformity probability=",p*100,"%")
        chisq, p  = stats.chisquare(quicksampler_cnt, ideal_cnt)
        print("QUICKSAMPLER/IDEAL chi2 uniformity probability=",p*100,"%")
        chisq, p  = stats.chisquare(uniform_cnt, ideal_cnt)
        print("UNIFORM/IDEAL chi2 uniformity probability=",p*100,"%")

        
        x, y_uniform = entropy_arr(uniform_map,10,7)
        x, y_unigen = entropy_arr(item["unigen_map"],10,7)
        x, y_diffusion = entropy_arr(item["diffusion_3-sat-unigen-500k_map"],10,7)
        x, y_quicksampler = entropy_arr(item["quicksampler_map"],10,7)
        
        print("  UNIGEN/UNIFORM CROSS ENTROPY=",binary_cross_entropy(y_unigen, y_uniform))
        print("  DIFFUSION/UNIFORM CROSS ENTROPY=",binary_cross_entropy(y_diffusion, y_uniform))
        print("  QUICKSAMPLER/UNIFORM CROSS ENTROPY=",binary_cross_entropy(y_quicksampler, y_uniform))

# show a legend on the plot 
    #plt.legend() 
  
    #print("BEFORE SHOW")
# function to show the plot 
    #plt.show() 
    #print("AFTER SHOW")

