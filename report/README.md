# CUDA Computer Vision

## Parallelization of Computer Vision and image Processing algorithms

### Guide: Prof. S. Gopalakrishnan

Images represent data in a 2d fashion and most image processing and computer vision algorithms involve processing and analysing blocks of images to attain desired results. These properties of image data and image processing algorithms make it a natural choice to parallelize them and exploit the SIMD nature of the algorithms. 


As a reference implementation, a naive sequential version of the algorithms has been implemented, which runs on the CPU. We also have run our algorithms using MATLAB and OpenCV and have provided a detailed performance analysis of the computation time taken. Below, we have explained explicitly the analysis and parallel implementation of 7 Image Processing algorithms.


## Algorithms Implemented 
* rgb2gray
* Histogram Equalization
* Gaussian Filtering
* Sobel Edge Detection
* Bilateral Filtering
* Median Filter
* k-Means Segmentation
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


### Bilateral Filtering

A bilateral filter is a non-linear, edge-preserving and noise-reducing smoothing filter for images. The intensity value at each pixel in an image is replaced by a weighted average of intensity values from nearby pixels.

#### Pseudo Code
``` python
# SIGMA_DOMAIN is the std dev of the filter in spatial domain
# SIGMA_RANGE is the std dev of the filter in intensity domain

for patch=cur in window(org):
  d = 2*kernel_size
  f   = (cur-org).^2
  cf  = expf(-(d*d)/(SIGMA_DOMAIN*SIGMA_DOMAIN*2))
  float sf  = expf(-(f*f)/(SIGMA_RANGE*SIGMA_RANGE*2))
  sum  = sum + cf*sf*cur
  sumk = sumk + cf*sf
end

sum = sum/sumk;
out[row*cols+col] = sum;
```

#### Speedup Comparison
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/report/bilateral_pa.png)

#### Parallelization
In Bilateral filtering, we use two kernels for convolution, one in Intensity domain and the other in the spatial domain.
A simple initial transfer to using the GPU, is done by taking the code for the individual pixel and use that as a basis for a kernel. This kernel is then launched with parameters which generates a thread for each pixel in the image. 

The values of the two dimensional Gaussian for the spatial difference can be pre computed, as it only depends on distances, and not the actual values of the pixels. A lot of the time running the kernel, is spent calculating the Gaussian of color intensities. CUDA allows for fast execution of hardware based implementations for a set of functions, at the price of a slight imprecision. 

#### Output Images
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/bilateral/taj_bil.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/bilateral/tiger_bil.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/bilateral/jet_bil.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/bilateral/lena_bil.jpg)

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

### Median Filtering
Median filtering is a non-linear, low-pass filtering method that can remove white or salt and pepper noise from an image or an audio signal.

#### Pseudo Code
``` C++
//CMF = Median Filter with CUDA 
__global__ void CMF(int* in, int* out, int c) {
  // k = input array index
  int k = c+blockDim.x*blockIdx.x+threadIdx.x;
  int M[N] = {0}; //major array
  int e[N] = {0}; //equal array
  int m[N] = {0}; //minor array
  //window index
  int i = k-c+threadIdx.y;
  for (int j=k-c; j<k-c+N; j++) {
    if (in[j]>in[i]) M[threadIdx.y]+=1;
    else if (in[j]<in[i]) m[threadIdx.y]+=1;
    else e[threadIdx.y]+=1;
  }
  for (int j = 0; j<N; j++) {
    if (M[j]==c || m[j]==c || e[j]>=c) {
      out[k] = in[k-c+j];
      return; 
    }
  }
}
```
#### Speedup Comparison
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/report/median_pa.png)

#### Parallelization
It requires sorting in the windows. So, in a sequential code, complexity is Nlog(L) (L is the window size)

The parallel implementation is simply dividing the image into subimages, one subimage for each thread. And the median filter in each of the threads, finally combining all of them into a single filtered image.

#### Output Images
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/input/aditi_pepper_input.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/median/aditi_median.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/input/taj_pepper_input.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/median/taj_median.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/input/tree_pepper_input.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/median/tree_median.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/input/jet_pepper_input.jpg)
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/output/median/jet_median.jpg)


### Stereo Vision
Dense Stereo Vision takes two input images, left and right, which are shifted and matched to generate the depth of each pixel. Non-occluding points are matched and the disparity for each point is computed and then the depth is retrieved.

#### Pseudo Code
``` python 
for k = 0 to MAX_SHIFT do
    Shift right image to the right by k pixels
    Perform Sum of Squared Differences (SSD) between left image and shifted right image
    Update the minSSD and disparity array.
            for each pixel coordinate (i,j) do
                if ssd(i,j) < minSSD(i,j) do
                    minSSD(i,j) <= ssd(i,j)
                    disparity(i,j) <= k 
                end
            end
end
```
#### Speedup Comparison
![rgb](https://github.com/yashbhalgat/ParallelCV/blob/master/report/stereo_pa.png)

#### Parallelization

The algorithm of stereo vision uses patch based mathcing across the image and hence this can be easily parallelized using CUDA blocks. We assign each CUDA block a range of pixels in a patch to compute the disparity. Since the disparity of a pixel is completely independent of the disparity at any other point, this algorithm is easily parallizable.   

#### Output Images
![Left Image](https://github.com/meetshah1995/EE-702/blob/master/stereo-vision/data/Aloe/view0.png)
![Right Image](https://github.com/meetshah1995/EE-702/blob/master/stereo-vision/data/Aloe/view1.png)

![NCC](https://github.com/meetshah1995/EE-702/blob/master/stereo-vision/data/Aloe/depthMultiOccularNCC_report.png)
![SAD](https://github.com/meetshah1995/EE-702/blob/master/stereo-vision/data/Aloe/depthMultiOccularSAD_report.png)
![SSD](https://github.com/meetshah1995/EE-702/blob/master/stereo-vision/data/Aloe/depthMultiOccularSSD_report.png)

## Speedup Achieved 

### CUDA CV vs OpenCV

| openCV v/s CUDA         | 0.48MP      | 2.5MP       | 3MP         | 8MP         | 12MP        |
|-------------------------|-------------|-------------|-------------|-------------|-------------|
| rgb2gray                | 3           | 6.666666667 | 6           | 4.5         | 4.133333333 |
| hisitogram equalisation | 2.5         | 6.25        | 5.555555556 | 5           | 4.591836735 |
| gaussian filtering      | 2.666666667 | 2.727272727 | 3.333333333 | 3.733333333 | 3.555555556 |
| Sobel Edge detection    | 8.333333333 | 15          | 12.5        | 6           | 8           |
| Bilateral filtering     | 6           | 11.5        | 11.42857143 | 11          | 8.875       |
| Median filter           | 7.142857143 | 10.95238095 | 10          | 10.57692308 | 10.14285714 |
| k-means segmentation    | 3           | 2.428571429 | 2.857142857 | 2.5         | 2.708333333 |
| stereo depth retreival  | 2.727272727 | 3.375       | 3.555555556 | 2.923076923 | 3.055555556 |

### CUDA CV vs MATLAB

| MATLAB v/s CUDA         | 0.48MP      | 2.5MP       | 3MP         | 8MP         | 12MP        |
|-------------------------|-------------|-------------|-------------|-------------|-------------|
| rgb2gray                | 2.5         | 5.666666667 | 5.6         | 4.2         | 3.866666667 |
| hisitogram equalisation | 1.2         | 3.2587      | 3.5487      | 4.563       | 3.876       |
| gaussian filtering      | 2.333333333 | 2.363636364 | 2.833333333 | 3.066666667 | 3.222222222 |
| Sobel Edge detection    | 7.936507937 | 14.28571429 | 11.9047619  | 5.714285714 | 7.619047619 |
| Bilateral filtering     | 4.958677686 | 9.504132231 | 9.445100354 | 9.090909091 | 7.334710744 |
| Median filter           | 5.330490405 | 8.173418621 | 7.462686567 | 7.893226177 | 7.569296375 |
| k-means segmentation    | 1.666666667 | 1.349206349 | 1.587301587 | 1.388888889 | 1.50462963  |
| stereo depth retreival  | 2.525252525 | 3.125       | 3.29218107  | 2.706552707 | 2.829218107 |


## References 
1: http://www.wseas.us/e-library/conferences/2011/Corfu/COMPUTERS/COMPUTERS-53.pdf

2: http://research.ijcaonline.org/volume88/number17/pxc3894051.pdf

3: http://users.eecs.northwestern.edu/~wkliao/Kmeans/

4: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4722322

5: http://cs.au.dk/~staal/dpc/20072300_paper_final.pdf



