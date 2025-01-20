import numpy as np
import time
from datetime import datetime

from config import Config
from utils.AllSolutions import AllSolutions
from utils.DimacsFile import DimacsFile

from satsolvers.QuickSampler import QuickSampler

from satuniformity.BenchmarksFile import BenchmarksFile
from satuniformity.UnigenSampler import UnigenSampler
from satuniformity.DiffusionSampler import DiffusionSampler
from satuniformity.QuickSampler import QuickSampler
import sys


# model_path = default = Config.train_dir + "/3-sat-unigen-500k"
# model_path = default=Config.train_dir + '/splot_500'
#model_path = default = Config.train_dir + "/diffusion-sat_24_02_18_22:26:16"
model_path = default = Config.train_dir + "/diffusion-sat_24_500k_steps_1m_train"
model_path = default = Config.train_dir +"/diffusion-sat_24_07_05_12:01:18" 
# ^^^ cosine
model_path = default = Config.train_dir +"/diffusion-sat_24_07_05_14:01:22"
# ^^^ cosine
model_path = default = Config.train_dir +"/diffusion-sat-167k"
dimacs_filename = "test0.dimacs"

np.set_printoptions(linewidth=2000, precision=3, suppress=True)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def dt2ms(dt):
    microseconds = time.mktime(dt.timetuple()) * 1000000 + dt.microsecond
    return int(round(microseconds / float(1000)))

def add_missing_keys(source_dict, target_dict, value=0):
    for key in source_dict:
        if not key in target_dict:
            target_dict[key] = value

def test_sk():
    print("TEST DIFFUSION BY SK")

    model_prefix = Config.train_dir +"/diffusion-sat-"
    
    #model_suffixes = [ "1k-fixed", "5k-fixed", "10k-fixed", "25k-fixed", "50k-fixed", "75k-fixed", "167k-fixed" ]
    model_suffixes = [ "1k-cos", "5k-cos", "10k-cos", "25k-cos", "50k-cos", "75k-cos", "167k-cos" ]
    #model_suffixes = [ "1k-fixed4", "5k-fixed4", "10k-fixed4", "25k-fixed4", "50k-fixed4", "75k-fixed4", "167k-fixed4" ]
    #model_suffixes = [ "1k-cos4", "5k-cos4", "10k-cos4", "25k-cos4", "50k-cos4", "75k-cos4", "167k-cos4" ]
    #steps_labels = [ "1K", "5K", "10K", "25K", "50K", "75K", "167K" ]

    # TODO: for cycle:
    model_suffix = model_suffixes[0]
    model_path = model_prefix+model_suffix

    df = DimacsFile(filename=dimacs_filename)
    df.load()
    all = AllSolutions(df.number_of_vars(), df.clauses())
    print("Counting # of solutions...")
    n_solutions = all.count()
    print("DIFF SOLUTIONS=",n_solutions)
    k = 50
    n_samples = n_solutions * k
#    n_samples = 50
    n_samples = n_samples * 10
    #n_samples = n_solutions *2
    print("Generating ",n_samples," samples using different samplers...")

    bf = BenchmarksFile()
    benchmark = bf.benchmarkFor(df.clauses())
    benchmark["n_solutions"] = n_solutions
    benchmark["n_samples"] = n_samples

    # >>> UNIGEN (must be the first, since we use keys from unigen_dict to fill missing keys in other samplers)
    print("unigen start")
    time1 = dt2ms(datetime.now())
    unigen_dict = UnigenSampler(df).samples(n_samples)
    time2 = dt2ms(datetime.now())
    print("unigen done")
    print("UnigenSampler generated ",len(unigen_dict)," distinct solutions.")

    benchmark["unigen_samples"]=sorted(unigen_dict.items())
    benchmark["unigen_speed"] = float(time2 - time1) / len(unigen_dict)

    # >>> DIFFUSION
    print("diffusion start")
    time1 = dt2ms(datetime.now())
    diffusion_dict = DiffusionSampler(model_path, dimacs_filename).samples(n_samples)
    time2 = dt2ms(datetime.now())
    print("diffusion done")
    print("DiffusionSampler generated ",len(diffusion_dict)," distinct solutions.")
    add_missing_keys(unigen_dict, diffusion_dict)
    benchmark["diffusion_samples"]=sorted(diffusion_dict.items())
    benchmark["diffusion_speed"] = float(time2 - time1) / len(diffusion_dict)


    # >>> QUICKSAMPLER
    print("quicksampler start")
    time1 = dt2ms(datetime.now())
    quicksampler_dict = QuickSampler(df).samples(n_samples)
    time2 = dt2ms(datetime.now())
    print("quicksampler done")
    add_missing_keys(unigen_dict, quicksampler_dict)
    benchmark["quicksampler_samples"]=sorted(quicksampler_dict.items())
    benchmark["quicksampler_speed"] = float(time2 - time1) / len(quicksampler_dict)

    print("n_samples: ", n_samples)
    print("unigen:   ", benchmark["unigen_samples"])
    print("diffusion:", benchmark["diffusion_samples"])
    print("quicksampler: ", benchmark["quicksampler_samples"])

    #bf.write(benchmark)


test_sk()
