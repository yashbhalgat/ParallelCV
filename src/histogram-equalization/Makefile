NVCC=nvcc
OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include
OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui
CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include
NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64
GCC_OPTS=-O3 -Wall -Wextra -m64

histeq: main.o kernel.o histeq_kernel.o loadSaveImage.o Makefile
	$(NVCC) -o histeq main.o kernel.o histeq_kernel.o loadSaveImage.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)

main.o: main.cpp timer.h utils.h
	g++ -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

histeq_kernel.o: histeq_kernel.cu loadSaveImage.h utils.h
	$(NVCC) -c histeq_kernel.cu -I $(OPENCV_INCLUDEPATH) $(NVCC_OPTS)

loadSaveImage.o: loadSaveImage.cpp loadSaveImage.h
	g++ -c loadSaveImage.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

kernel.o: kernel.cu utils.h
	$(NVCC) -c kernel.cu $(NVCC_OPTS)

clean:
	rm -f *.o hw
	find . -type f -name '*.exr' | grep -v memorial | xargs rm -f