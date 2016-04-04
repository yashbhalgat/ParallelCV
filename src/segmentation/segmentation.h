#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstring>
#include <cuda_runtime.h>

using namespace std;

const int RANGE_MAX = 255;
const int RANGE_MIN = 0;
const double INF = 9999.0;
const int MAX_ITER = 20;
const float OMEGA = 0.72;
const float EPSILON = 0.0005;
const float c1 = 1.49;
const float c2 = 1.49;
const int DATA_DIM = 3;

struct data
{
    int info[DATA_DIM];
};

struct particle
{
    data *position;
    data *pBest;
    data *velocity;
};

struct GBest
{
    short *gBestAssign;
    data *centroids;
    int *arrCentroids;
};

float getRandom(float low, float high);
float getRandomClamped();
float getDistance(data first, data second);
float fitness(const short *assignMat, const data *datas, const data *centroids, 
              int data_size, int cluster_size);
void assignDataToCentroid(short *assignMat, const data *datas, 
                          const data *centroids, int data_size, 
                          int cluster_size);
void initializePSO(particle *particles, GBest& gBest, const data *datas, 
                   int data_size, int particle_size, int cluster_size);
GBest hostPsoClustering(data *datas, int data_size, int particle_size, 
                        int cluster_size, int max_iter);
extern "C" float devFitness(short* assignMat, int* datas, int* centroids, 
                            int data_size, int cluster_size);
extern "C" GBest devicePsoClustering(data *datas, int *flatDatas, int data_size, 
                                     int particle_size, int cluster_size, 
                                     int max_iter);
#endif /* SEGMENTATION_H */
