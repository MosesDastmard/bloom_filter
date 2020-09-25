-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->

# Bloom Filter

A Bloom filter is a space-efficient probabilistic data structure, that is used to test whether an element is a member of a set. False positive matches are possible, but false negatives are not – in other words, a query returns either "possibly in set" or "definitely not in set". Elements can be added to the set, but not removed (though this can be addressed with the counting Bloom filter variant); the more items added, the larger the probability of false positives.

Bloom proposed the technique for applications where the amount of source data would require an impractically large amount of memory if "conventional" error-free hashing techniques were applied. He gave the example of a hyphenation algorithm for a dictionary of 500,000 words, out of which 90% follow simple hyphenation rules, but the remaining 10% require expensive disk accesses to retrieve specific hyphenation patterns. With sufficient core memory, an error-free hash could be used to eliminate all unnecessary disk accesses; on the other hand, with limited core memory, Bloom's technique uses a smaller hash area but still eliminates most unnecessary accesses. For example, a hash area only 15% of the size needed by an ideal error-free hash still eliminates 85% of the disk accesses.
<a href='https://arxiv.org/pdf/1803.04189.pdf'>Read More...</a>

## Hashing task!
The bloom filter is an array initialized with all False of lenght hash_size, a huge prime number.
To be sure that the probability of false positive is small enough we take $p\approx\big(1-e^{\frac{-kn}{m}}\big)^k \approx 10^{-15}$.
To do this, knowing that the numbers of passwords in passwords1.txt is $n=10^8$, we pick the follow values for hash_size $m=10000000019$ a huge prime number base on the chapter 1 of [book](https://books.google.it/books?id=ONU4tfT_GxcC&dq=algorithm+Umesh+Vazirani&hl=en&sa=X&ved=0ahUKEwiC1KPghormAhXIG5oKHZ_gBqkQ6AEIKTAA) titled Algorithm and hash_funcs_num $k=20$.

We convert each password to an array corresponding ordinals of the length of the password.

$$\forall\: \text{hash function} \: h_j \:\: \text{with} \: j \in \{0, \dots ,k-1\} \:\:\:\: h_j = \sum_{i=0}^{19} ord_i C_{i,j} \: mod \: m $$

Each $C_{i,j}$ is taken randomly from a uniform discret distribution ranging from $0$ to $m-1$

To have a faster numerical computation we use numba package that can convert python function to be compiled on C/C++ that has the ability to run the function on GPU and CPU(Parallel).


```python
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
```


```python
# The class keep the information about bloom filter like number of hash functions (K), bloom filter size (m)
# length of the passwords to make the coefficient in hash functions, and store the bloom filter itself.  
class hash_configuration():
    def __init__(self, name, hash_size, string_size = 20, hash_funcs_num = 10):
        self.name = name
        self.string_size = string_size
        self.hash_size = hash_size
        self.hash_funcs_num = hash_funcs_num
        self.coef = np.random.randint(0,self.hash_size-1,(self.hash_funcs_num, self.string_size), dtype = np.int64)
```


```python
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
```


```python
hash_size = 10000000019 #It is a huge prime number
string_size = 20
hash_funcs_num = 20
bloom_filter_conf = hash_configuration('bloom filter', hash_size, string_size, hash_funcs_num)
```


```python
BloomFilter('passwords1.txt', 'passwords2.txt', bloom_filter_conf)
```

    100000000it [14:03, 118514.84it/s]
    39000000it [05:47, 112384.28it/s]
    

    Number of hash function used:  20
    Number of duplicates detected:  14000000
    Probability of false positives:  1.467177285944451e-15
    Execution time:  1191.0610814094543  secs
    

As you can see above this algorithm takes less than 20 minutes to insert the passwords in the bloom filter and check the duplicates 


```python
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
```


```python
duplicate('passwords1.txt', 'passwords2.txt')
```

    Execution time:  715.9290888309479  secs
    The number of exact duplicated passwords is:  14000000
    

Thanks to this function using Map/Reduce (PySpark), we can evaluate the exact number of duplicated passwords inside the two .txt files in less than 12 minutes.
As we can see this number is equal to the one we get from the bloom filter so it means that there isn't any false positive.
