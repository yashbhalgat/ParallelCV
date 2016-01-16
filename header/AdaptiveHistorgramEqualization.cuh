/*
 * DeviceOperation.cuh
 *
 *      Author: meetshah1995
 */

#ifndef DEVICEOPERATION_CUH_
#define DEVICEOPERATION_CUH_

	template <typename T>
	__global__ void deviceOperation(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *resultRed, unsigned char *resultGreen, unsigned char *resultBlue, int width, int height, T *filter) {
		(*filter)(red, green, blue, resultRed, resultGreen, resultBlue, width, height);
	}

#endif /* DEVICEOPERATION_CUH_ */
