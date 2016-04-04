file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (gaussian_filter_files main.cpp)
CUDA_ADD_EXECUTABLE(gaussian_filter ${gaussian_filter_files} ${hdr} ${cu})