#include <opencv2/opencv.hpp>
#include "segmentation.h"

int main(int argc, char** argv)
{
    srand(time(NULL));
    IplImage* inputImage = NULL;
    inputImage = cvLoadImage("../../input/Stony_beach.jpg",-1);

    int width = inputImage->width;
    int height = inputImage->height;
    int channel = inputImage->nChannels;

    char *arrImage = new char[width * height * channel];
    int *flatDatas = new int[width * height * channel];
    data *datas = new data[width * height];

    // Load image to array
    for (int i = 0; i < width * height; i++)
    {
        arrImage[i * channel + 0] = 
            (unsigned char) inputImage->imageData[i * channel + 0];
        arrImage[i * channel + 1] = 
            (unsigned char) inputImage->imageData[i * channel + 1];
        arrImage[i * channel + 2] = 
            (unsigned char) inputImage->imageData[i * channel + 2];

        flatDatas[i * channel + 0] = 
            (unsigned char) inputImage->imageData[i * channel + 0];
        flatDatas[i * channel + 1] = 
            (unsigned char) inputImage->imageData[i * channel + 1];
        flatDatas[i * channel + 2] = 
            (unsigned char) inputImage->imageData[i * channel + 2];

        data d;

        d.info[0] = (unsigned char) inputImage->imageData[i * channel + 0];
        d.info[1] = (unsigned char) inputImage->imageData[i * channel + 1];
        d.info[2] = (unsigned char) inputImage->imageData[i * channel + 2];

        datas[i] = d;
    }

    // PSO parameters
    int particle_num, cluster_num, max_iter;
    char comp;

    cout << "Number of cluster : ";
    cin >> cluster_num;
    cout << "Number of particle : ";
    cin >> particle_num;
    cout << "Number of iteration : ";
    cin >> max_iter;

    do
    {
        cout << "CPU or GPU (C / G) ? : ";
        cin >> comp;
    }
    while(comp != 'C' && comp != 'G');

    cout << endl;

    clock_t begin = clock();

    GBest gBest;

    // Check if use host code or device code
    if(comp == 'C')
        gBest = hostPsoClustering(datas, width * height, particle_num, 
                                  cluster_num, max_iter);
    else
        gBest = devicePsoClustering(datas, flatDatas, width * height, 
                                    particle_num, cluster_num, max_iter);

    clock_t end = clock();

    cout << "Time elapsed : " << (double)(end - begin) / CLOCKS_PER_SEC << "s" 
         << endl;

    // Compute quantization error of clusters, less is better
    if(comp == 'C')
    {
        cout << "Quantization Error : " 
             << fitness(gBest.gBestAssign, datas, gBest.centroids, 
                        width * height, cluster_num) 
             << endl;
    }
    else
    {
        cout << "Quantization Error : " 
             << devFitness(gBest.gBestAssign, flatDatas, gBest.arrCentroids, 
                width * height, cluster_num) 
             << endl;       
    }

    // List for cluster color
    unsigned char colorList[9][3] = { { 0, 0, 255 }, { 255, 0, 0 }, 
                                      { 0, 255, 0 }, { 255, 255, 0 }, 
                                      { 255, 0, 255 }, { 255, 128, 128 }, 
                                      { 128, 128, 128 }, { 128, 0, 0 }, 
                                      { 255, 128, 0 } };

    // Coloring clusters
    for (int i = 0; i < width * height; i++)
    {
        for (int j = 0; j < cluster_num; j++)
        {
            if (gBest.gBestAssign[i] == j)
            {
                arrImage[i * channel + 0] = colorList[j][0];
                arrImage[i * channel + 1] = colorList[j][1];
                arrImage[i * channel + 2] = colorList[j][2];
            }
        }
    }

    if(comp == 'C')
        delete[] gBest.centroids;
    else
        cudaFreeHost(gBest.arrCentroids);

    delete[] gBest.gBestAssign;

    // Write array to image
    IplImage* outImage = cvCreateImage(cvSize(width, height), 
                                       inputImage->depth, channel);
    outImage->imageData = arrImage;

    cvSaveImage("out1.jpg", outImage);

    cvNamedWindow("Result");
    cvShowImage("Result", outImage);
    cvWaitKey(0);

    // Cleanup
    cvReleaseImage(&inputImage);
    cvReleaseImage(&outImage);

    delete[] arrImage;

    return 0;
}