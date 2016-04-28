NVCC=nvcc
OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include
OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui
CUDA_INCLUDEPATH=/usr/local/cuda-5.5/include
CUDA_HELPER_FILES=/usr/local/cuda-5.5/samples/common/inc/
NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64
GCC_OPTS=-O3 -Wall -Wextra -m64

median_filter: main.o median.o Makefile
	g++ `pkg-config --cflags --libs opencv` -L/usr/local/cuda/lib64 -lcudart -o median_filter main.o median.o

main.o: main.cpp
	g++ -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDEPATH)

median.o: median.cu
	nvcc -c median.cu $(NVCC_OPTS) -I $(CUDA_HELPER_FILES)
	
clean:
	rm -f *.o *.png median_filter
