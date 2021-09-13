from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = load_iris()

k = list(range(1,11))
iner = [KMeans(n_clusters = i, random_state=0).fit(data.data).inertia_ for i in k]

plt.plot(k,iner)
plt.annotate('elbow',xy=(2,iner[2-1]),xytext=(3,iner[2-1]+100),arrowprops=dict(arrowstyle="->"))
plt.annotate('elbow',xy=(3,iner[3-1]),xytext=(3,iner[2-1]+100),arrowprops=dict(arrowstyle="->"))
plt.annotate(f'(2,{round(iner[2-1],2)})',xy=(2,iner[1]),xytext=(0.8,100))
plt.annotate(f'(3,{round(iner[3-1],2)})',xy=(3,iner[2]),xytext=(3,20))

plt.savefig('elbow.png')