cmake_minimum_required( VERSION 3.11 )

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()
if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
if (POLICY CMP0069)
    cmake_policy(SET CMP0069 NEW)
endif (POLICY CMP0069)
if (POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
endif (POLICY CMP0072)

project( smplxpp )
option( SMPLX_USE_FFAST_MATH "Enable ffast-math compiler flag, may cause numerical problems" ON )

set( INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include" )
set( SRC_DIR "${PROJECT_SOURCE_DIR}/src" )
set( VENDOR_DIR "${PROJECT_SOURCE_DIR}/3rdparty" )
set( CNPY_DIR "${VENDOR_DIR}/cnpy" )
set( ZLIB_DIR "${VENDOR_DIR}/zlib" )

set( CMAKE_CXX_STACK_SIZE "10000000" )
set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON)
set( CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules" )

set( PROJ_NAME "smplxpp" )

if( ${SMPLX_USE_FFAST_MATH} )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast" )
endif()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD /GLT /Ox")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MDd")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MD /GLT")
# Silence deprecation warning coming from Eigen; silence secure warning
add_definitions(-D_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING -D_CRT_SECURE_NO_WARNINGS)


file(GLOB_RECURSE SOURCES ${SRC_DIR}/*.cpp)
file(GLOB_RECURSE HEADERS ${INCLUDE_DIR}/smplx/*.hpp)

set( VENDOR_SOURCES ${CNPY_DIR}/cnpy.cpp)

# ensure M_PI, etc available
add_definitions( -DGLEW_STATIC -D_USE_MATH_DEFINES )

set( DEPENDENCIES )

include_directories(
    ${VENDOR_DIR}/misc  
    ${CNPY_DIR}
)

# Eigen
if(NOT DEFINED EIGEN3_INCLUDE_DIR) 
    find_package(Eigen3 QUIET)
    if(NOT Eigen3_FOUND)
        set(EIGEN3_INCLUDE_DIR $ENV{EIGEN3_INCLUDE_DIR})
        if(NOT EIGEN3_INCLUDE_DIR)
            message(FATAL_ERROR "EIGEN3 not found in system path or environment variable")
        endif()
    else()
        set(EIGEN3_INCLUDE_DIR ${Eigen3_INCLUDE_DIRS})  
    endif()
endif()
include_directories(${EIGEN3_INCLUDE_DIR})

# Zlib
find_package(ZLIB QUIET)
if (ZLIB_FOUND)
    message( STATUS "Using system zlib" )
    set( DEPENDENCIES ${DEPENDENCIES} ZLIB::ZLIB )
else()
    message( STATUS "Using included zlib." )
    add_subdirectory(${ZLIB_DIR})
    include_directories(${ZLIB_DIR})
    set( DEPENDENCIES ${DEPENDENCIES} zlibstatic )
endif()

add_library( ${PROJ_NAME} STATIC ${SOURCES} ${VENDOR_SOURCES} )
set_target_properties( ${PROJ_NAME} PROPERTIES OUTPUT_NAME "smplx" )
target_include_directories(${PROJ_NAME} PUBLIC
    $<BUILD_INTERFACE:${INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>   
)
target_link_libraries( ${PROJ_NAME} ${DEPENDENCIES} )
install(TARGETS ${PROJ_NAME} DESTINATION lib)

if(WIN32)  
    add_definitions(-DNOMINMAX)
elseif(CYGWIN)
elseif(APPLE)
elseif(UNIX)
    target_link_libraries( ${PROJ_NAME} -pthread )
    target_link_libraries( example -pthread )
    add_definitions(-DUNIX)
    add_compile_options(-Wno-deprecated-declarations)
    add_compile_options(-Wno-unused-result)
endif(WIN32)

# create version.hpp
configure_file( version.hpp.in "${PROJECT_BINARY_DIR}/include/smplx/version.hpp" )

source_group( "Header Files" FILES ${HEADERS} )
source_group( "Source Files" FILES ${SOURCES} )
source_group( "Vendor Source Files" FILES ${VENDOR_SOURCES} )