# CUDA Computer Vision

## Parallelization of Computer Vision and image Processing algorithms

Images represent data in a 2d fashion and most image processing and computer vision algorithms involve processing and analysing blocks of images to attain desired results. These properties of image data and image processing algorithms make it a natural choice to parallelize them and exploit the SIMD nature of the algorithms. 


As a reference implementation, a naive sequential version of the algorithms has been implemented, which runs on the CPU. We also have run our algorithms using MATLAB and OpenCV and have provided a detailed performance analysis of the computation time taken. Below, we have explained explicitly the analysis and parallel implementation of 7 Image Processing algorithms.

## Group members 

* Meet Pragnesh Shah [ 13D070003 ]
* Yash Bhalgat [ 13D070014 ]

## Algorithms Implemented 
* rgb2gray
* Histogram Equalization
* k-Means Segmentation
* Gaussian Filtering
* Sobel Edge Detection
* Median Filter
* Depth Retrieval using Stereo Vision
`
### rgb2gray

RGB to grayscale conversion is the most basic pixel-wise conversion. In this we take a pixel in the RBG domain and multiply each color intensity with respective constants and assign the sum as the intensiity of the grayscale pixel. 

#### Pseudo Code
``` python
for pixel in image:
  output_pixel = pixel_green * 0.587 + pixel_blue * 0.114 + pixel_red * 0.299
end 
```

#### Parallelization

In this case parallelism is attained by simply dividing the image into equal sub-matrices for each thread. The kernel pseudocode is as given above.

#### Speedup Comparison
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/report/bilateral_pa.png)

#### Input and Output Images
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/input/aditi.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/rgb2gray/aditi_gray.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/input/taj.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/rgb2gray/taj_gray.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/input/tiger.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/rgb2gray/tiger_gray.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/input/jet.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/rgb2gray/jet_gray.jpg)

### Histogram Equalization

Histogram equalization is a method in image processing of contrast adjustment using the image's histogram. The histogram for intensity values is obtained and then this histogram is equalized in order to generally increase the global contrast of the image under consideration.

For some images, such as remote sensing image, because the gray distribution is in a relatively narrow range, resulting in image details are not clear enough, and have lower contrast. Therefore, we need to do histogram equalization processing for these images.

#### Pseudo Code
``` python

# Calculate histogram for input image  
max_gray_level = max(max(image))
T[0] = histogram[0]

for k in range(1,max_gray_level):
	T[k] = T[k-1] + histogram[k]  
end  

for r  in range(0,max_gray_level):
	S[r] = round(T[r]*max_gray_level)  
end

# Generate equlized image from the histogram
for y in range(0,M):  
	for x in range(0,N):  
		r = image[y, x]  
		output[y, x] = S[r]  
	end  
end  
```

#### Parallelization
Suppose the image has N levels (intensity levels). Each thread processes one gray-level.
Each block will process L levels, hence has L threads. 


  ![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/report/histeq1.png)
  ![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/report/histeq2.png)

Because of the size limit of shared memory, we can’t first copy data from global memory to shared memory, and each thread must read data from global memory. After threads finish computing, each thread has a sub-histogram that size is L. And then, each block need to reduce its sub-histograms to a bigger sub-histogram.


#### Speedup Comparison
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/report/histogram_equalization_pa.png)

#### Input & Output Images
![rgb](https://raw.githubusercontent.com/yashbhalgat/ParallelCV/master/src/histogram-equalization/memorial_raw_large.png)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/histogram-equalization/memorial_histeq.png)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/bilateral/tiger_bil.pgm)

### Gaussian Filtering

The idea of Gaussian smoothing is to use this 2-D distribution as a `point-spread' function, and this is achieved by convolution. Since the image is stored as a collection of discrete pixels we need to produce a discrete approximation to the Gaussian function before we can perform the convolution. The kernel is dependent on the patch size and the parameters for the Gaussian function which lead to various levels of smoothening. 

#### Pseudo Code 
``` python
patch_size = N*N

patch_i_j = patchCenteredAt(image(i,j))
for patch_i_j in image:
  output(i,j) = conv(gaussianKernel(mean,sd) , patch_i_j)
  output(i,j) = normazlize(output(i,j))
end
``` 
#### Speedup Comparison
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/report/gaussian_filtering_pa.png)

#### Parallelization

As we can see from the pseudo code the entire algorithm is involves 2D convolution of a fixed kernel with overlapping patches that span the entire input. These properties of the patch filtering help us in exploiting the shared memory of blocks in CUDA (where essentially a patch corresponds to a block). We inlined the kernel so as to prevent its computation for every block. We observed the highest speedup for this algorithm.  

#### Output Images
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/input/aditi.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/gaussian/aditi_gaussian.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/input/taj.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/gaussian/taj_gaussian.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/input/tiger.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/gaussian/tiger_gaussian.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/input/jet.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/gaussian/jet_gaussian.jpg)

### K-means Segmentation

k-means segmentation is a method of vector quantization which aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. This results in a partitioning of the image into Voronoi cells.

#### Pseudo Code
``` python
k = number_of_clusters

# Initialize centroids randomly
centroids = getRandomCentroids(k)

iterations = 0
oldCentroids = None

while not (iterations > MAX_ITERATIONS or oldCentroids == centroids ):
    oldCentroids = centroids
    iterations += 1
    
    # Assign labels to each datapoint based on centroids
    labels = getLabels(image, centroids)
    
    # Assign centroids based on datapoint labels
    centroids = getCentroids(image, labels, k)
end
``` 

#### Speedup Comparison
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/report/histogram_equalization_pa.png)

#### Parallelization

#### Output Images