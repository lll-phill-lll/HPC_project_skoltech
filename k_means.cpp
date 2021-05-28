#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;

typedef vector<double> Point;
typedef vector<Point> Points;
typedef vector<double> OptPoints;

struct Compare {
    double val;
    size_t index;
};
#pragma omp declare reduction(minimum : struct Compare : omp_out = omp_in.val < omp_out.val ? omp_in : omp_out)

#pragma omp declare reduction(vec_plus_size_t: vector<size_t> : \
                              transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), plus<size_t>())) \
                    initializer(omp_priv = omp_orig)


#pragma omp declare reduction(vec_plus_double: vector<double> : \
                              transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), plus<double>())) \
                    initializer(omp_priv = omp_orig)

// Gives random number in range [0..max_value]
unsigned int UniformRandom(unsigned int max_value) {
    unsigned int rnd = ((static_cast<unsigned int>(rand()) % 32768) << 17) |
                       ((static_cast<unsigned int>(rand()) % 32768) << 2) | rand() % 4;
    return ((max_value + 1 == 0) ? rnd : rnd % (max_value + 1));
}

double Distance(const Point& point1, const Point& point2) {
    double distance_sqr = 0;
    size_t dimensions = point1.size();
    for (size_t i = 0; i < dimensions; ++i) {
        distance_sqr += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return distance_sqr;
}


size_t FindNearestCentroid(const Points& centroids, const Point& point) {
    double min_distance = Distance(point, centroids[0]);
    size_t centroid_index = 0;
    for (size_t i = 1; i < centroids.size(); ++i) {
        double distance = Distance(point, centroids[i]);
        if (distance < min_distance) {
            min_distance = distance;
            centroid_index = i;
        }
    }
    return centroid_index;
}

size_t FindNearestCentroid2D(const OptPoints& centroids, double a, double b) {
    double min_distance = ((a - centroids[0]) * (a - centroids[0])) + ((b - centroids[1]) * (b - centroids[1]));
    size_t centroid_index = 0;
    for (size_t i = 1; i < centroids.size() / 2; ++i) {
        double distance = ((a - centroids[i * 2]) * (a - centroids[i * 2])) + ((b - centroids[i * 2 + 1]) * (b - centroids[i * 2 + 1]));
        if (distance < min_distance) {
            min_distance = distance;
            centroid_index = i;
        }
    }
    return centroid_index;
}




// Calculates new centroid position as mean of positions of 3 random centroids
Point GetRandomPosition(const Points& centroids) {
    size_t K = centroids.size();
    int c1 = rand() % K;
    int c2 = rand() % K;
    int c3 = rand() % K;
    size_t dimensions = centroids[0].size();
    Point new_position(dimensions);
    for (size_t d = 0; d < dimensions; ++d) {
        new_position[d] = (centroids[c1][d] + centroids[c2][d] + centroids[c3][d]) / 3;
    }
    return new_position;
}

vector<size_t> KMeans(const Points& data, size_t K) {
    size_t data_size = data.size();
    size_t dimensions = data[0].size();
    vector<size_t> clusters(data_size);

    // Initialize centroids randomly at data points
    Points centroids(K);
    for (size_t i = 0; i < K; ++i) {
        centroids[i] = data[UniformRandom(data_size - 1)];
    }

    bool converged = false;
    while (!converged) {
        converged = true;
        #pragma omp parallel for reduction(&:converged)
        for (size_t i = 0; i < data_size; ++i) {
            size_t nearest_cluster = FindNearestCentroid(centroids, data[i]);
            if (clusters[i] != nearest_cluster) {
                clusters[i] = nearest_cluster;
                converged = false;
            }
        }
        if (converged) {
            break;
        }
        vector<size_t> clusters_sizes(K);
        centroids.assign(K, Point(dimensions));

        #pragma omp parallel
        {
            Points thread_centroids(K);
            vector<size_t> thread_clusters_sizes(K);
            thread_centroids.assign(K, Point(dimensions));
            #pragma omp for nowait reduction(vec_plus_size_t: clusters_sizes)
            for (size_t i = 0; i < data_size; ++i) {
                for (size_t d = 0; d < dimensions; ++d) {
                    thread_centroids[clusters[i]][d] += data[i][d];
                }
                clusters_sizes[clusters[i]] += 1;
            }

            for (size_t i = 0; i < K; ++i) {
                for (size_t d = 0; d < dimensions; ++d) {
                    #pragma omp atomic
                    centroids[i][d] += thread_centroids[i][d];
                }
            }
        }
        for (size_t i = 0; i < K; ++i) {
            if (clusters_sizes[i] != 0) {
                for (size_t d = 0; d < dimensions; ++d) {
                    centroids[i][d] /= clusters_sizes[i];
                }
            }
        }
        for (size_t i = 0; i < K; ++i) {
            if (clusters_sizes[i] == 0) {
                centroids[i] = GetRandomPosition(centroids);
            }
        }
    }

    return clusters;
}


vector<size_t> KMeans2D(const OptPoints& data_opt, size_t K) {
    size_t data_size_normal = data_opt.size() / 2;
    size_t dimensions = 2;
    size_t data_size = data_opt.size();
    vector<size_t> clusters(data_size_normal);

    // Initialize centroids randomly at data points
    OptPoints centroids(2 * K);
    for (size_t i = 0; i < K; ++i) {
        size_t num = UniformRandom(data_size_normal - 1);
        centroids[2 * i] = data_opt[2 * num];
        centroids[2 * i + 1] = data_opt[2 * num + 1];
    }

    bool converged = false;
    while (!converged) {
        converged = true;
        #pragma omp parallel for reduction(&:converged)
        for (size_t i = 0; i < data_size_normal; ++i) {
            size_t nearest_cluster = FindNearestCentroid2D(centroids, data_opt[2 * i], data_opt[2 * i + 1]);
            if (clusters[i] != nearest_cluster) {
                clusters[i] = nearest_cluster;
                converged = false;
            }
        }
        // std::cout << "not\n";
        if (converged) {
            // std::cout << "yes\n";
            break;
        }
        vector<size_t> clusters_sizes(K);
        centroids.assign(2 * K, 0.0);

        #pragma omp parallel
        {
            #pragma omp for nowait reduction(vec_plus_size_t: clusters_sizes) reduction(vec_plus_double: centroids)
            for (size_t i = 0; i < data_size_normal; ++i) {
                double dot1 = data_opt[2 * i];
                double dot2 = data_opt[2 * i + 1];
                centroids[clusters[i] * 2] += dot1;
                centroids[clusters[i] * 2 + 1] += dot2;
                ++clusters_sizes[clusters[i]];
            }
        }
        for (size_t i = 0; i < K; ++i) {
            if (clusters_sizes[i] != 0) {
                centroids[2 * i] /= clusters_sizes[i];
                centroids[2 * i + 1] /= clusters_sizes[i];
            }
        }
        for (size_t i = 0; i < K; ++i) {
            if (clusters_sizes[i] == 0) {
                int c1 = rand() % K;
                int c2 = rand() % K;
                int c3 = rand() % K;
                double pos1 = (centroids[2 * c1] + centroids[2 * c2] + centroids[2 * c3]) / 3;
                double pos2 = (centroids[2 * c1 + 1] + centroids[2 * c2 + 1] + centroids[2 * c3 + 1]) / 3;
                centroids[2 * i] = pos1;
                centroids[2 * i + 1] = pos2;
            }
        }
    }

    return clusters;
}

// void ReadPoints(Points* data, ifstream& input) {
//     size_t data_size;
//     size_t dimensions;
//     input >> data_size >> dimensions;
//     data->assign(data_size, Point(dimensions));
//     for (size_t i = 0; i < data_size; ++i) {
//         for (size_t d = 0; d < dimensions; ++d) {
//             double coord;
//             input >> coord;
//             (*data)[i][d] = coord;
//         }
//     }
// }

// optimized version using atof()
void ReadPoints(Points* data, OptPoints* data_opt, size_t& dimensions, ifstream& input) {
    size_t data_size;
    input >> data_size >> dimensions;
    if (dimensions != 2) {
        data->assign(data_size, Point(dimensions));
        string s;
        for (size_t i = 0; i < data_size; ++i) {
            for (size_t d = 0; d < dimensions; ++d) {
                input >> s;
                (*data)[i][d] = atof(s.c_str());
            }
        }
    } else {
        string s;
        for (size_t i = 0; i < data_size * 2; ++i) {
            input >> s;
            data_opt->push_back(atof(s.c_str()));
        }
    }
}

void WriteOutput(const vector<size_t>& clusters, ofstream& output) {
    for (size_t i = 0; i < clusters.size(); ++i) {
        output << clusters[i] << '\n';
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::printf("Usage: %s number_of_clusters input_file output_file\n", argv[0]);
        return 1;
    }
    size_t K = atoi(argv[1]);

    char* input_file = argv[2];
    ifstream input;
    input.open(input_file, ifstream::in);
    if (!input) {
        cerr << "Error: input file could not be opened\n";
        return 1;
    }
    OptPoints data_opt;
    Points data;
    size_t dim;
    ReadPoints(&data, &data_opt, dim, input);
    input.close();

    char* output_file = argv[3];
    ofstream output;
    output.open(output_file, ifstream::out);
    if (!output) {
        cerr << "Error: output file could not be opened\n";
        return 1;
    }

    srand(123); // for reproducible results
    vector<size_t> clusters;
    auto start = chrono::high_resolution_clock::now();
    if (dim != 2) {
        clusters = KMeans(data, K);
    } else {
        clusters = KMeans2D(data_opt, K);
    }
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "Time taken by function: " << duration.count()  / 1000 << " seconds " << duration.count() % 1000 << " milliseconds" << endl;
    WriteOutput(clusters, output);
    output.close();

    return 0;
}
