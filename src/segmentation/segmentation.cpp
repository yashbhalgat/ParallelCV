#include "segmentation.h"

float getRandom(float low, float high)
{
    return low + float(((high - low) + 1) * rand() / ((float) RAND_MAX + 1));
}

float getRandomClamped()
{
    return (float) rand() / (double) RAND_MAX;
}

/*
 * Round real number to its nearest integer
 */
int round(float x)
{
    if (x > 0.0)
        return (int) floor(x + 0.5);
    else
        return (int) ceil(x - 0.5);
}

/*
 * Get euclidean distance between 2 pixels
 */
float getDistance(data first, data second)
{
    double total = 0.0;

    for (int i = 0; i < DATA_DIM; i++)
    {
        int res = (first.info[i] - second.info[i]);
        total += res * res;
    }

    return sqrt(total);
}

/*
 * Get error for given centroids
 */
float fitness(const short *assignMat, const data *datas, const data *centroids, 
              int data_size, int cluster_size)
{
    double total = 0.0;

    for (int i = 0; i < cluster_size; i++)
    {
        double subtotal = 0.0;

        for (int j = 0; j < data_size; j++)
        {
            if (assignMat[j] == i)
                subtotal += getDistance(datas[j], centroids[i]);
        }

        total += subtotal / data_size;
    }

    return total / cluster_size;
}

/*
 * Assign pixels to centroids
 */
void assignDataToCentroid(short *assignMat, const data *datas, 
                          const data *centroids, int data_size, 
                          int cluster_size)
{
    for (int i = 0; i < data_size; i++)
    {
        int nearestCentroidIdx = 0;
        double nearestCentroidDist = INF;

        for (int j = 0; j < cluster_size; j++)
        {
            double nearestDist = getDistance(datas[i], centroids[j]);

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
void initializePSO(particle *particles, GBest& gBest, const data *datas, 
                   int data_size, int particle_size, int cluster_size)
{
    for (int i = 0; i < particle_size; i++)
    {
        particle p;

        p.pBest = new data[cluster_size];
        p.position = new data[cluster_size];
        p.velocity = new data[cluster_size];

        particles[i] = p;

        for (int j = 0; j < cluster_size; j++)
        {
            data d;

            for (int k = 0; k < DATA_DIM; k++)
                d.info[k] = 0;

            particles[i].velocity[j] = d;

            int rand = round(getRandom(0, data_size - 1));

            particles[i].position[j] = datas[rand];
            particles[i].pBest[j] = datas[rand];
        }
    }

    gBest.centroids = new data[cluster_size];

    for (int j = 0; j < cluster_size; j++)
    {
        data d;

        for (int k = 0; k < DATA_DIM; k++)
            d.info[k] = round(abs(getRandom(RANGE_MIN, RANGE_MAX)));

        gBest.centroids[j] = d;
    }
}

GBest hostPsoClustering(data *datas, int data_size, int particle_size, 
                        int cluster_size, int max_iter)
{
    // initialize
    GBest gBest;
    particle *particles = new particle[particle_size];
    short **assignMatrix = new short*[particle_size];
    short **pBestAssign = new short*[particle_size];
    gBest.gBestAssign = new short[data_size];

    for (int i = 0; i < particle_size; i++)
    {
        assignMatrix[i] = new short[data_size];
        pBestAssign[i] = new short[data_size];

        for (int j = 0; j < data_size; j++)
        {
            assignMatrix[i][j] = 0;
            pBestAssign[i][j] = 0;
        }
    }

    for (int i = 0; i < data_size; i++)
        gBest.gBestAssign[i] = 0;

    initializePSO(particles, gBest, datas, data_size, particle_size, 
                  cluster_size);

    // Iteration
    for (int i = 0; i < max_iter; i++)
    {
        float rp = getRandomClamped();
        float rg = getRandomClamped();

        // foreach particle
        for (int j = 0; j < particle_size; j++)
        {
            // foreach dimension
            for (int k = 0; k < cluster_size; k++)
            {
                // foreach data dimension
                for (int l = 0; l < DATA_DIM; l++)
                {
                    particles[j].velocity[k].info[l] = 
                        round(OMEGA * particles[j].velocity[k].info[l]
                        + c1 * rp * (particles[j].pBest[k].info[l] - 
                            particles[j].position[k].info[l])
                        + c2 * rg * (gBest.centroids[k].info[l] - 
                            particles[j].position[k].info[l]));

                    particles[j].position[k].info[l] += 
                        particles[j].velocity[k].info[l];
                }
            }

            assignDataToCentroid(assignMatrix[j], datas, particles[j].position, 
                                 data_size, cluster_size);
        }

        for (int j = 0; j < particle_size; j++)
        {
            if (fitness(assignMatrix[j], datas, particles[j].position, 
                        data_size, cluster_size)
                < fitness(pBestAssign[j], datas, particles[j].pBest, data_size, 
                          cluster_size))
            {
                for (int k = 0; k < cluster_size; k++)
                    particles[j].pBest[k] = particles[j].position[k];

                // assignDataToCentroid(pBestAssign[j], datas, 
                // particles[j].pBest, data_size, cluster_size);
                for(int k = 0; k < data_size; k++)
                    pBestAssign[j][k] = assignMatrix[j][k];

                if (fitness(pBestAssign[j], datas, particles[j].pBest, 
                            data_size, cluster_size)
                    < fitness(gBest.gBestAssign, datas, gBest.centroids, 
                              data_size, cluster_size))
                {
                    for (int k = 0; k < cluster_size; k++)
                        gBest.centroids[k] = particles[j].pBest[k];

                    // assignDataToCentroid(gBest.gBestAssign, datas, 
                    // gBest.centroids, data_size, cluster_size);
                    for(int k = 0; k < data_size; k++)
                        gBest.gBestAssign[k] = pBestAssign[j][k];
                }
            }
        }
    }

    // cleanup
    for (int i = 0; i < particle_size; i++)
    {
        delete[] assignMatrix[i];
        delete[] pBestAssign[i];
        delete[] particles[i].pBest;
        delete[] particles[i].position;
        delete[] particles[i].velocity;
    }

    delete particles;
    delete[] assignMatrix;
    delete[] pBestAssign;

    return gBest;
}
