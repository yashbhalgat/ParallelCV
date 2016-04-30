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
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/report/rgb2gray_pa.png)

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
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/gaussian-filtering/aditi_gaussian.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/gaussian-filtering/taj_gaussian.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/gaussian-filtering/tiger_gaussian.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/gaussian-filtering/jet_gaussian.jpg)


### Sobel Edge Detection

The algorithm uses two 3×3 kernels which are convolved with the original image to calculate approximations of the derivatives - one for horizontal changes, and one for vertical. If we define A as the source image, and Gx and Gy are two images which at each point contain the horizontal and vertical derivative approximations respectively, the computation for edges can be easily done using this vertical and horizontal derivatives and comparison with a threshold.

Sobel operator:
`𝐻𝑥 = [ −1 0 1 , −2 0 2 , −1 0 1 ]`
`𝐻𝑦 =  [ −1 −2 −1 , 0 0 0 , 1 2 1 ]`

To save computation time, we find  `GM (𝑥, 𝑦) = | 𝐻𝑥 | + | 𝐻𝑦 |`  (instead of sum of squares)
The local edge strength is computed using this quantity.

#### Speedup Comparison
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/report/sobel_pa.png)

#### Parallelization
The input image is divided into subimages and each subimage is passed to the blocks.
The sub-matrix is stored in the shared memory. A gaussian filter is applied on the subimage.
Then edge detection is applied on each of these sub-matrices.

![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/report/sobel.png)

#### Output Images
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/sobel/beach_sob.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/sobel/taj_sob.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/sobel/tiger_sob.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/sobel/jet_sob.jpg)


### Median Filtering

#### Pseudo Code

#### Speedup Comparison

#### Parallelization

#### Output Images
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/sobel/beach_sob.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/sobel/taj_sob.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/sobel/tiger_sob.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/sobel/jet_sob.jpg)


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
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/report/kmeans_segmentation_pa.png)

#### Parallelization

Features vector of each pixel RGBXY are taken. In this case, descriptor in p(i, j) is P(r,g,b, x, y). This representation leads to a segmentation based on both, the color and position, of the pixel in the image.
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/report/kmeans.jpg)

Parallel k-means: The optimization process seek to parallelize the nearest centroid, because there are no dependencies between one iteration to another. On the other hand, the computation between centers remains serial, since each iteration depends on the centers that have been calculated in previous iterations. Furthermore, it may include calculating on-site of the new centers, if done by the average, at the same time, by calculating the belonging to each group.

![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/report/kmeans2.png)

We run a global sum matching in order to obtain the global clusters after the parallel part is complete. We however couldn't get results as good as the serial implementation using this technique.


#### Output Images
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/segment/aditi_segment.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/segment/beach_segment.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/segment/tiger_segment.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/segment/jet_segment.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/segment/taj_segment.jpg)

## References 
