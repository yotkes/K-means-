import mykmeanssp as ckm
import argparse
import pandas as pd
import numpy as np
import sys

parser = argparse.ArgumentParser()

parser.add_argument("k", type=int, help="Number of centroids - integer >=1")
if len(sys.argv) == 5:
    parser.add_argument("MAX_ITER", type=int, help="Maximum number of iterations to reach convergence - integer >=1")
elif len(sys.argv) == 4:
    MAX_ITER = 300
else:
    print("K is missing")
    exit(1)

parser.add_argument("filename1", type=str, help="Name of file1")
parser.add_argument("filename2", type=str, help="Name of file2")
args = parser.parse_args()
k = args.k

if len(sys.argv) == 5:
    MAX_ITER = args.MAX_ITER

filename1 = args.filename1
filename2 = args.filename2

df1 = pd.read_csv(filename1, header=None)
df2 = pd.read_csv(filename2, header=None)
df = pd.merge(df1, df2, on=0,sort=True).drop(0,1)
N = len(df)
d = len(df.columns)

if ((k>=N) or (d<=0) or (k<=0) or (N<=0) or (MAX_ITER<=0)):
    print("invalid inputs")
    exit(0)

def calculate_difference(vector, centroid):
    sum = 0
    for j in range(d):
        sum += ((vector[j] - centroid[j])**2)
    return sum

def check_min_distance(vector, centroids):
    dist = calculate_difference(vector,centroids[0])
    for cluster in range(1, len(centroids)):
        new_dist = calculate_difference(vector, centroids[cluster])
        if (new_dist<dist):
            dist = new_dist
    return dist

def k_means_pp(obs):
    np.random.seed(0)
    rand = np.random.choice(N, 1)
    indexes = [rand[0]]
    centroids = [obs[rand[0]]]
    for j in range(1,k):
        dists = [check_min_distance(obs[i], centroids) for i in range(N)]
        sums = sum(dists)
        probs = [dists[i]/sums for i in range(N)]
        rand = np.random.choice(N,1,p=probs)
        centroids.append(obs[rand[0]])
        indexes.append(rand[0])

    rep = ""
    for i in indexes:
        rep += str(i)
        rep += ","
    print(rep[:-1])
    return indexes


obs = df.to_numpy()
first = k_means_pp(obs)
to_c_obs = obs.tolist()
to_send = [k, N, d, MAX_ITER, first, to_c_obs]
list_c = ckm.fit(to_send)
list_c = np.array(list_c)
list_c = np.round(list_c,4)


i = 0
for element in list_c.flat:
    i=i+1
    print(element, end='')
    if i == d:
        i=0
        print("\n", end='')
        continue
    print(",", end='')


