Commands to be run if you don't want to use the make file:

g++ -c main.cpp
nvcc -arch=sm_20 -I/usr/local/cuda-5.5/samples/common/inc/ -c median.cu
g++ `pkg-config --cflags --libs opencv` -L/usr/local/cuda/lib64 -lcudart -o median main.o median.o
