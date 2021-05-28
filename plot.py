import numpy as np
import matplotlib.pyplot as plt


def read_input():
    xs = []
    ys = []
    clusters = []

    input_file = open("input.txt", "r")
    was_header = False

    for line in input_file:
        if not was_header:
            was_header = True
            continue

        coords = line.split()
        xs.append(float(coords[0]))
        ys.append(float(coords[1]))

    return np.array(xs), np.array(ys)


def read_clusters():
    clusters = []

    input_file = open("clusters.txt", "r")

    for line in input_file:
        clusters.append(int(line))

    clusters = np.array(clusters)
    return clusters, len(np.unique(clusters))


xs, ys = read_input()
clusters, num_clusters = read_clusters()
for i in range(num_clusters):
    plt.scatter(xs[clusters==i], ys[clusters==i], s=10)
plt.show()
