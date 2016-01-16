/*
 * CUDA-CV.cpp
 *
 *      Author: meetshah1995
 */

#include "../header/CUDA_CVImage.h"

CUDA_CVImage::CUDA_CVImage(std::string imageSource) {
	image = imread(imageSource);

	imageSize = sizeof(uchar)*image.rows*image.cols;

	bChannel = new uchar [imageSize];
	gChannel = new uchar [imageSize];
	rChannel = new uchar [imageSize];

	d_bChannel = NULL;
	d_gChannel = NULL;
	d_rChannel = NULL;

	d_bResultChannel = NULL;
	d_gResultChannel = NULL;
	d_rResultChannel = NULL;

	for(int i = 0;i < image.cols ;i++) {
		for(int j = 0;j < image.rows ;j++) {
			Vec3b intensity = image.at<Vec3b>(j,i);

			bChannel[j*image.cols+i] = intensity.val[0];
			gChannel[j*image.cols+i] = intensity.val[1];
			rChannel[j*image.cols+i] = intensity.val[2];
	    }
	}
}

CUDA_CVImage::~CUDA_CVImage() {
	if(!bChannel) {
		delete bChannel;
	}
	if(!gChannel) {
		delete bChannel;
	}
	if(!rChannel) {
		delete bChannel;
	}
	if(!d_bChannel) {
		cudaFree(d_bChannel);
	}
	if(!d_gChannel) {
		cudaFree(d_gChannel);
	}
	if(!d_rChannel) {
		cudaFree(d_rChannel);
	}
	if(!d_bResultChannel) {
		cudaFree(d_bResultChannel);
	}
	if(!d_gResultChannel) {
		cudaFree(d_gResultChannel);
	}
	if(!d_rResultChannel) {
		cudaFree(d_rResultChannel);
	}
}

void CUDA_CVImage::initDeviceMemoryForImage() {
	if(d_bChannel == NULL) {
		cudaMalloc((void**)&d_bChannel, imageSize);
	}
	if(d_gChannel == NULL) {
		cudaMalloc((void**)&d_gChannel, imageSize);
	}
	if(d_rChannel == NULL) {
		cudaMalloc((void**)&d_rChannel, imageSize);
	}
	if(d_bResultChannel == NULL) {
		cudaMalloc((void**)&d_bResultChannel, imageSize);
	}
	if(d_gResultChannel == NULL) {
		cudaMalloc((void**)&d_gResultChannel, imageSize);
	}
	if(d_rResultChannel == NULL) {
		cudaMalloc((void**)&d_rResultChannel, imageSize);
	}
}

void CUDA_CVImage::freeDeviceMemory() {
	if(!d_bChannel) {
		cudaFree(d_bChannel);
	}
	if(!d_gChannel) {
		cudaFree(d_gChannel);
	}
	if(!d_rChannel) {
		cudaFree(d_rChannel);
	}
	if(!d_bResultChannel) {
		cudaFree(d_bResultChannel);
	}
	if(!d_gResultChannel) {
		cudaFree(d_gResultChannel);
	}
	if(!d_rResultChannel) {
		cudaFree(d_rResultChannel);
	}
}

uchar* CUDA_CVImage::getRHostChannel() {
	return rChannel;
}
uchar* CUDA_CVImage::getGHostChannel() {
	return gChannel;
}
uchar* CUDA_CVImage::getBHostChannel() {
	return bChannel;
}
uchar* CUDA_CVImage::getRDeviceChannel() {
	return d_rChannel;
}
uchar* CUDA_CVImage::getGDeviceChannel() {
	return d_gChannel;
}
uchar* CUDA_CVImage::getBDeviceChannel() {
	return d_bChannel;
}
uchar* CUDA_CVImage::getRResultDeviceChannel() {
	return d_rResultChannel;
}
uchar* CUDA_CVImage::getGResultDeviceChannel() {
	return d_gResultChannel;
}
uchar* CUDA_CVImage::getBResultDeviceChannel() {
	return d_bResultChannel;
}

void CUDA_CVImage::copyImageToDeviceMemory() {
	cudaMemcpy(d_bChannel, bChannel, imageSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_gChannel, gChannel, imageSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rChannel, rChannel, imageSize, cudaMemcpyHostToDevice);
}

void CUDA_CVImage::copyImageFromResultToHostMemory() {
	cudaMemcpy(bChannel, d_bResultChannel, imageSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(gChannel, d_gResultChannel, imageSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(rChannel, d_rResultChannel, imageSize, cudaMemcpyDeviceToHost);

	//Create also new version of image
	for(int i = 0;i < getCols() ;i++) {
		for(int j = 0;j < getRows();j++) {
			image.at<Vec3b>(j, i) = Vec3b(getBHostChannel()[j*getCols()+i], getGHostChannel()[j*getCols()+i], getRHostChannel()[j*getCols()+i]);
	    }
	}
}

int CUDA_CVImage::getRows() {
	return image.rows;
}
int CUDA_CVImage::getCols() {
	return image.cols;
}

void CUDA_CVImage::saveImage(std::string imageName) {
	imwrite(imageName, image);
}
void CUDA_CVImage::showImage(int delay) {
	imshow("CUDA_CVImage", image);
	waitKey(delay);
}

