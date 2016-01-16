/*
 * IPNCLImage.h
 *
 *      Author: kebolt
 */
#ifndef IPNCLIMAGE_H_
#define IPNCLIMAGE_H_

#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned char uchar;
using namespace cv;

class IPNCLImage {
public:
	IPNCLImage(std::string imageSource);
	virtual ~IPNCLImage();

	void initDeviceMemoryForImage();
	void freeDeviceMemory();
	void copyImageToDeviceMemory();
	void copyImageFromResultToHostMemory();
	void saveImage(std::string imageName);
	void showImage(int howLong);

	int getRows();
	int getCols();

	uchar* getRHostChannel();
	uchar* getGHostChannel();
	uchar* getBHostChannel();
	uchar* getRDeviceChannel();
	uchar* getGDeviceChannel();
	uchar* getBDeviceChannel();
	uchar* getRResultDeviceChannel();
	uchar* getGResultDeviceChannel();
	uchar* getBResultDeviceChannel();


private:
	Mat image;
	uchar *rChannel;
	uchar *gChannel;
	uchar *bChannel;

	uchar *d_rChannel;
	uchar *d_gChannel;
	uchar *d_bChannel;

	uchar *d_rResultChannel;
	uchar *d_gResultChannel;
	uchar *d_bResultChannel;

	int imageSize;
};

#endif /* IPNCLIMAGE_H_ */
