#!/usr/bin/python

import argparse
from collections import defaultdict
from functools import partial
import numpy as np
from scipy import misc
from scipy.cluster.vq import kmeans, vq
from time import time

# param in_file - name of the input file
# param k - number of clusters
# return time to come to convergence
# rtype float
def scipy_kmeans(in_file, k, max_iter=20, threshold=1e-5):
    img = misc.imread(in_file)
    data = img.reshape((-1,3)).astype(float)

    start = time()
    centroids, _ = kmeans(data, k, iter=max_iter, thresh=threshold)
    code, _ = vq(data, centroids)
    res = centroids[code]
    elapsed = time() - start

    res2 = res.reshape((img.shape))
    misc.imsave('scipy_k=%d_%s'%(k, in_file), res2)

    return elapsed

def diy_kmeans(in_file, k, max_iter=20, threshold=1e-5):
    img = misc.imread(in_file)
    data = img.reshape((-1,3)).astype(float)

    start = time()
    centroids = data[np.random.choice(data.shape[0], k)]
    RSS = float('inf')
    iteration = 0
    while True:
        # print 'Iteration: %d, RSS: %f'%(iteration, RSS)
        buckets = defaultdict(list)
        n_RSS = 0
        for pixel in data:
            dists = map(lambda c: np.linalg.norm(pixel-c), centroids)
            ind = np.argmin(dists)
            n_RSS += dists[ind]
            buckets[ind].append(pixel)
        centroids = map(partial(np.mean, axis=0), buckets.values())
        if RSS - n_RSS < threshold or iteration > max_iter:
            break
        RSS = n_RSS
        iteration += 1
    res = np.array([centroids[np.argmin(map(lambda c: np.linalg.norm(pixel-c), centroids))] for pixel in data])

    elapsed = time() - start

    res2 = res.reshape((img.shape))
    misc.imsave('diy_k=%d_%s'%(k, in_file), res2)

    return elapsed


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', help="input file", type=str)
    parser.add_argument('k', help='number of clusters', type=int)
    args = parser.parse_args()

    scipy_time = scipy_kmeans(args.in_file, args.k)
    diy_time = diy_kmeans(args.in_file, args.k)

    print 'Time (scipy): %f'%scipy_time
    print 'Time (diy): %f'%diy_time
