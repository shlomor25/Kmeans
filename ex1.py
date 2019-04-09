import numpy as np
from scipy.misc import imread
from init_centroids import init_centroids


def load_image(path):
    A = imread(path)
    A_norm = A.astype(float) / 255.
    img_size = A_norm.shape
    X = A_norm.reshape(img_size[0] * img_size[1], img_size[2])
    return X


def print_iter(i, centroids):
    print("iter {}:".format(i), end=' ')
    for j in range(k):
        r = np.floor(centroids[j][0] * 100) / 100
        g = np.floor(centroids[j][1] * 100) / 100
        b = np.floor(centroids[j][2] * 100) / 100
        if r == 0: r = "0."
        if g == 0: g = "0."
        if b == 0: b = "0."
        # print centroids (RGB)
        print("[{}, {}, {}]".format(r, g, b), end='')
        if j != (k - 1): print(", ", end='')
    print()


# Euclidean Distance Calculator
def distance(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


def KMeans(X, k):
    # init centroids and clusters
    C = init_centroids(X, k)
    clusters = np.zeros(len(X))
    # 10 iterations
    for b in range(11):
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            distances = distance(X[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        # Finding the new centroids by taking the average value
        print_iter(b, C)
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)


if __name__ == "__main__":
    K = [2, 4, 8, 16]
    X = load_image('dog.jpeg')
    for k in K:
        print("k=%s:"%k)
        KMeans(X, k)
