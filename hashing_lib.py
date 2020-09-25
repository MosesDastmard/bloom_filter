#%%
import numpy as np
# numba is the package that can convert python function to be compiled on C/C++
# that has the ability to run the function in on GPU and CPU(Parallel) that help
# numerical computation to be done faster. REALLY FASTERRRRRRR 
from numba import jit
from tqdm import tqdm
import time
import math
# PySpark is the package to run Map/Reduce procedure over Spark in python
from pyspark import SparkContext
#%%
# The class keep the information about bloom filter like number of hash functions (K), bloom filter size (m)
# length of the passwords to make the coefficient in hash functions, and store the bloom filter itself.  
class hash_configuration():
    def __init__(self, name, hash_size, string_size = 20, hash_funcs_num = 10):
        self.name = name
        self.string_size = string_size
        self.hash_size = hash_size
        self.hash_funcs_num = hash_funcs_num
        self.coef = np.random.randint(0,self.hash_size-1,(self.hash_funcs_num, self.string_size), dtype = np.int64)

############## Python is Faster while communicates with C/C++ #################
# The below function gives the ordinals related to each passwords and computes the hash values based on
# coefficients that are stored on bloom filter configuration
@jit(nopython=True, nogil=True, parallel=True, cache=True)
def parallel(x,coef,hash_size):
    return (x*coef).sum(axis=1)%hash_size
# This function gives as input a passwords as string, computes the ordinals as list of integers and 
# call the parallel function to generate the hash values  
def hash_gen(line, conf):
    x = np.array(list(map(ord, line[0:-1])))
    return parallel(x,conf.coef,conf.hash_size)

# It initialize the bloom filter with all false values, streams the passwords1.txt, gets the hash values, updates
# the bloom filter, checks the duplicated passwords in passwords2.txt 
def BloomFilter(passwords1 , passwords2 , bloom_filter_conf):
    start_time = time.time()
    bloom_filter = np.zeros(bloom_filter_conf.hash_size, dtype=bool)
    num_inserted_pass = 0
    # streaming passwords1.txt
    with open(passwords1) as f:
        for line in tqdm(f):
            bloom_filter[hash_gen(line = line, conf = bloom_filter_conf)] = True        
            num_inserted_pass += 1
    duplicated_pass_num = 0
    passes_num = 0
    # streaming passwords2.txt
    with open(passwords2) as f:
        for line in tqdm(f):
            passes_num += 1
            # check duplication
            if all(bloom_filter[hash_gen(line = line, conf = bloom_filter_conf)]):
                duplicated_pass_num += 1
    end_time = time.time()
    print('Number of hash function used: ', bloom_filter_conf.hash_funcs_num)
    print('Number of duplicates detected: ', duplicated_pass_num)
    k = bloom_filter_conf.hash_funcs_num
    m = bloom_filter_conf.hash_size
    n = num_inserted_pass
    e = math.exp
    # The probability of false positive in theory 
    print('Probability of false positives: ', (1-e(-k*n/m))**k)
    print('Execution time: ', end_time - start_time, ' secs')
    

#%%
################### PySpark: parallel computing on single machine ####################
# In order to compute exact number of false positive we start with computing 
# the true number of duplicated passwords inside the two .txt files
# the difference between duplicated number extrated from bloom filter and true number of duplicated
# passwords is the exact number of real false positive
# in this regard the below varibales are defined:
# pass1_num: number of unique passwords in passwords1.txt
# pass2_num: number of unique passwords in passwords2.txt
# union_num: number of unique passwords in passwords1.txt and passwords2.txt
# the pass2_num - (union_num - pass1_num) will give the true number of duplicated passwords
def duplicate(passwords1 , passwords2):
    start_time = time.time()
    sc = SparkContext('local[*]', 'Find dupicated passwords')
    pass1_rdd = sc.textFile(passwords1)
    pass1_num = pass1_rdd.distinct().count()
    pass2_rdd = sc.textFile(passwords2)
    pass2_num = pass2_rdd.distinct().count()
    dup_rdd = pass2_rdd.union(pass1_rdd).distinct()
    union_num = dup_rdd.count()
    end_time = time.time()
    print('Execution time: ', end_time - start_time, ' secs')
    print('The number of exact duplicated passwords is: ', pass2_num - (union_num - pass1_num))