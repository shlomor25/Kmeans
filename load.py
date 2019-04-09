import numpy as np
from scipy.misc import imread
from init_centroids import init_centroids
import matplotlib.pyplot as plt


def print_image(path, centroids):
    k = len(centroids)
    A = imread(path)
    A_norm = A.astype(float) / 255.
    img_size = A_norm.shape
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            min_c = 0
            min_dst = float('inf')
            for l in range(k):
                dst = dist(A_norm[i][j], centroids[l])
                if dst < min_dst:
                    min_c = l
                    min_dst = dst
            A_norm[i][j] = centroids[min_c]
    plt.imshow(A_norm)
    plt.grid(False)
    plt.show()


def load_image(path):
    A = imread(path)
    A_norm = A.astype(float) / 255.
    img_size = A_norm.shape
    X = A_norm.reshape(img_size[0] * img_size[1], img_size[2])
    return X


def dist(a, b):
    return np.sqrt(((a[0] - b[0])**2) + ((a[1] - b[1])**2) + ((a[2] - b[2])**2))


def printIter(i, centroidsList):
    print("iter {}:".format(i), end=' ')
    for j in range(k):
        r = np.floor(centroidsList[j][0] * 100) / 100
        g = np.floor(centroidsList[j][1] * 100) / 100
        b = np.floor(centroidsList[j][2] * 100) / 100
        print("[{}, {}, {}]".format(r, g, b), end='')
        if j != (k - 1):
            print(", ", end='')
    print()


def distance(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


def divide(X, k):
    C = init_centroids(X, k)
    clusters = np.zeros(len(X))
    # Loop will run till the error becomes zero
    for b in range(11):
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            distances = distance(X[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        # Finding the new centroids by taking the average value
        printIter(b, C)
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)
    print_image('dog.jpeg', C)


if __name__ == "__main__":
    image_path = 'dog.jpeg'
    X = load_image(image_path)
    K = [2, 4, 8, 16]
    for k in K:
        print("k=%s:"%k)
        divide(X, k)