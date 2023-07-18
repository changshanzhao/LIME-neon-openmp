
cmake_minimum_required(VERSION 2.8.0)
project(lime_processing)

set(CMAKE_CXX_STANDARD 14)

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )


set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )  

find_package( OpenCV 3.1.0 REQUIRED )

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories( 
    ${OpenCV_INCLUDE_DIRS} 

)
include_directories( 
    ${OpenMP_INCLUDE_DIRS} 

)


include_directories( ${PROJECT_SOURCE_DIR}/include )

add_subdirectory( src )
add_subdirectory( test )








