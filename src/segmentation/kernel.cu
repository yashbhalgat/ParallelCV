#include <math_functions.h>
#include "segmentation.h"

/*
 * Get euclidean distance between 2 pixels
 */
__host__ __device__ 
float devGetDistance(int *first, int *second)
{
    float total = 0.0f;

    for (int i = 0; i < DATA_DIM; i++)
    {
        int res = (first[i] - second[i]);
        total += res * res;
    }

    return sqrt(total);
}

/*
 * Get error for given centroids
 */
__host__ __device__ 
float devFitness(short* assignMat, int* datas, int* centroids, int data_size, 
                 int cluster_size)
{
    float total = 0.0f;

    for (int i = 0; i < cluster_size; i++)
    {
        float subtotal = 0.0f;

        for (int j = 0; j < data_size; j++)
        {
            if (assignMat[j] == i)
                subtotal += devGetDistance(&datas[j * DATA_DIM], 
                                           &centroids[i * DATA_DIM]);
        }

        total += subtotal / data_size;
    }

    return total / cluster_size;
}

/*
 * Assign pixels to centroids
 */
__host__ __device__ 
void devAssignDataToCentroid(short *assignMat, int *datas, int *centroids, 
                             int data_size, int cluster_size)
{
    for (int i = 0; i < data_size; i++)
    {
        int nearestCentroidIdx = 0;
        float nearestCentroidDist = INF;

        for (int j = 0; j < cluster_size; j++)
        {
            float nearestDist = devGetDistance(&datas[i * DATA_DIM], 
                                               &centroids[j * DATA_DIM]);

            if (nearestDist < nearestCentroidDist)
            {
                nearestCentroidDist = nearestDist;
                nearestCentroidIdx = j;
            }
        }

        assignMat[i] = nearestCentroidIdx;
    }
}

/*
 * Initialize necessary variables for PSO
 */
void initialize(int *positions, int *velocities, int *pBests, int *gBest, 
                const data* datas, int data_size, int particle_size, 
                int cluster_size)
{
    for (int i = 0; i < particle_size * cluster_size * DATA_DIM; i+= DATA_DIM)
    {
        int rand = round(getRandom(0, data_size - 1));

        for(int j = 0; j < DATA_DIM; j++)
        {
            positions[i + j] = datas[rand].info[j];
            pBests[i + j] = datas[rand].info[j];
            velocities[i + j] = 0;
        }
    }

    for(int i = 0; i < cluster_size * DATA_DIM; i++)
        gBest[i] = pBests[i];
}

/*
 * Kernel to update particle
 */
__global__ void kernelUpdateParticle(int *positions, int *velocities, 
                                     int *pBests, int *gBest, short *posAssign, 
                                     int* datas, float rp, float rg, 
                                     int data_size, int particle_size, 
                                     int cluster_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= particle_size * cluster_size * DATA_DIM)
        return;

    // Update particle velocity and position
    velocities[i] = (int)lroundf(OMEGA * velocities[i] 
                    + c1 * rp * (pBests[i] - positions[i])
                    + c2 * rg * 
                        (gBest[i % (cluster_size * DATA_DIM)] - positions[i]));

    positions[i] += velocities[i];
}

/*
 * Kernel to update particle
 */
__global__ void kernelUpdatePBest(int *positions, int *pBests, short *posAssign, 
                                  short *pBestAssign, int* datas, int data_size, 
                                  int particle_size, int cluster_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetParticle = i * cluster_size * DATA_DIM;
    int offsetAssign = i * data_size;

    if(i >= particle_size)
        return;

    devAssignDataToCentroid(&posAssign[offsetAssign], datas, 
                            &positions[offsetParticle], data_size, cluster_size);

    // Update pBest
    if (devFitness(&posAssign[offsetAssign], datas, &positions[offsetParticle], 
                   data_size, cluster_size)
        < devFitness(&pBestAssign[offsetAssign], datas, &pBests[offsetParticle], 
                     data_size, cluster_size))
    {
        // Update pBest position
        for (int k = 0; k < cluster_size * DATA_DIM; k++)
            pBests[offsetParticle + k] = positions[offsetParticle + k];

        // Update pBest assignment matrix
        for(int k = 0; k < data_size; k++)
            pBestAssign[offsetAssign + k] = posAssign[offsetAssign + k];
    }
}

/*
 * Wrapper to initialize and running PSO on device
 */
extern "C" GBest devicePsoClustering(data *datas, int *flatDatas, int data_size, 
                                     int particle_size, int cluster_size, 
                                     int max_iter)
{
    // Initialize host memory
    int *positions = new int[particle_size * cluster_size * DATA_DIM];
    int *velocities = new int[particle_size * cluster_size * DATA_DIM];
    int *pBests = new int[particle_size * cluster_size * DATA_DIM];
    int *gBest = new int[cluster_size * DATA_DIM];
    short *posAssign = new short[particle_size * data_size];
    short *pBestAssign = new short[particle_size * data_size];
    short *gBestAssign = new short[data_size];

    // Initialize assignment matrix to cluster 0
    for(int i = 0; i < particle_size * data_size; i++)
    {
        posAssign[i] = 0;
        pBestAssign[i] = 0;

        if(i < data_size)
            gBestAssign[i] = 0;
    }

    initialize(positions, velocities, pBests, gBest, datas, data_size, 
               particle_size, cluster_size);

    // Initialize device memory
    int *devPositions, *devVelocities, *devPBests, *devGBest;
    short *devPosAssign, *devPBestAssign;
    int *devDatas;

    size_t size = sizeof(int) * particle_size * cluster_size * DATA_DIM;
    size_t assign_size = sizeof(short) * particle_size * data_size;

    cudaMalloc((void**)&devPositions, size);
    cudaMalloc((void**)&devVelocities, size);
    cudaMalloc((void**)&devPBests, size);
    cudaMalloc((void**)&devGBest, sizeof(int) * cluster_size * DATA_DIM);
    cudaMalloc((void**)&devPosAssign, assign_size);
    cudaMalloc((void**)&devPBestAssign, assign_size);
    cudaMalloc((void**)&devDatas, sizeof(int) * data_size * DATA_DIM);

    // Copy data from host to device
    cudaMemcpy(devPositions, positions, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devVelocities, velocities, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devPBests, pBests, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devGBest, gBest, sizeof(int) * cluster_size * DATA_DIM, 
               cudaMemcpyHostToDevice);
    cudaMemcpy(devPosAssign, posAssign, assign_size, cudaMemcpyHostToDevice);
    cudaMemcpy(devPBestAssign, pBestAssign, assign_size, cudaMemcpyHostToDevice);
    cudaMemcpy(devDatas, flatDatas, sizeof(int) * data_size * DATA_DIM, 
               cudaMemcpyHostToDevice);

    // Threads and blocks number
    int threads = 32;
    int blocksPart = (particle_size / threads) + 1;
    int blocksFull = (particle_size * cluster_size * DATA_DIM / threads) + 1;

    // Iteration
    for (int iter = 0; iter < max_iter; iter++)
    {        
        float rp = getRandomClamped();
        float rg = getRandomClamped();

        kernelUpdateParticle<<<blocksFull, threads>>>
            (devPositions, devVelocities, devPBests, devGBest, devPosAssign, 
             devDatas, rp, rg, data_size, particle_size, cluster_size);

        kernelUpdatePBest<<<blocksPart, threads>>>
            (devPositions, devPBests, devPosAssign, devPBestAssign, devDatas, 
             data_size, particle_size, cluster_size);

        // Compute gBest on host
        cudaMemcpy(pBests, devPBests, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(pBestAssign, devPBestAssign, assign_size, 
                   cudaMemcpyDeviceToHost);

        for(int i = 0; i < particle_size; i++)
        {
            // Get slice of array
            int offsetParticle = i * cluster_size * DATA_DIM;
            int offsetAssign = i * data_size;

            // Compare pBest and gBest
            if (devFitness(&pBestAssign[offsetAssign], flatDatas, 
                           &pBests[offsetParticle], data_size, cluster_size)
                < devFitness(gBestAssign, flatDatas, gBest, data_size, 
                             cluster_size))
            {
                // Update gBest position
                for (int k = 0; k < cluster_size * DATA_DIM; k++)
                    gBest[k] = pBests[offsetParticle + k];

                // Update gBest assignment matrix
                for(int k = 0; k < data_size; k++)
                    gBestAssign[k] = pBestAssign[offsetAssign + k];
            }
        }

        cudaMemcpy(devGBest, gBest, sizeof(int) * cluster_size * DATA_DIM, 
                   cudaMemcpyHostToDevice);
    }

    // Copy gBest from device to host
    cudaMemcpy(gBest, devGBest, sizeof(int) * cluster_size * DATA_DIM, 
               cudaMemcpyDeviceToHost);

    // Cleanup
    delete[] positions;
    delete[] velocities;
    delete[] pBests;
    delete[] posAssign;
    delete[] pBestAssign;

    cudaFree(devPositions);
    cudaFree(devVelocities);
    cudaFree(devPBests);
    cudaFree(devGBest);
    cudaFree(devPosAssign);
    cudaFree(devPBestAssign);
    cudaFree(devDatas);

    GBest gBestReturn;
    gBestReturn.gBestAssign = gBestAssign;
    gBestReturn.arrCentroids = gBest;

    return gBestReturn;
}