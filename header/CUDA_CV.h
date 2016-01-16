/*
 * IPNCL.cuh
 *
 *      Author: meetshah1995
 */

#ifndef IPNCL_CUH_
#define IPNCL_CUH_

#include "../header/IPNCLImage.h"
#include "../header/DeviceOperation.cuh"

namespace IPNCL {

	template <typename T>
	void filter(IPNCLImage *image, T *oper) {
		image->initDeviceMemoryForImage();
		image->copyImageToDeviceMemory();

		dim3 block(16, 16);
		dim3 grid(image->getCols()/16+1, image->getRows()/16+1);

		/*
		 * Copy filter object to device memory.
		 */
		T *d_operation;
		cudaMalloc((void**)&d_operation, sizeof(T));
		cudaMemcpy((void*)d_operation, (void*)oper, sizeof(T), cudaMemcpyHostToDevice);

		deviceOperation<T><<<grid, block>>>(image->getRDeviceChannel(), image->getGDeviceChannel(), image->getBDeviceChannel(), image->getRResultDeviceChannel(), image->getGResultDeviceChannel(), image->getBResultDeviceChannel(), image->getCols(), image->getRows(), d_operation);
		cudaDeviceSynchronize();

		image->copyImageFromResultToHostMemory();
		image->freeDeviceMemory();
		cudaFree(d_operation);
		delete oper;
	}
}


#endif /* IPNCL_CUH_ */
