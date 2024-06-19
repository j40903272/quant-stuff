cmake_minimum_required(VERSION 3.13)

project(HFT
    VERSION 0.1
    LANGUAGES CXX
)

set(CMAKE_C_COMPILER "gcc-13")
set(CMAKE_CXX_COMPILER "g++")

add_definitions(-DPROTOBUF_INLINE_NOT_IN_HEADERS=0)

set(default_build_type "Release")

if(NOT CMAKE_BUILD_TYPE)
    message(WARNING "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
endif()

message("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
IF(CMAKE_BUILD_TYPE MATCHES Debug)
    message("debug mode")
    add_definitions(-DAlphaOneDebug)
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)
IF(CMAKE_BUILD_TYPE MATCHES Release)
    message("release mode")
    remove_definitions(-DAlphaOneDebug)
ENDIF(CMAKE_BUILD_TYPE MATCHES Release)

IF(ALPHA_PERFECT)
    message("Only build AlphaPerfect")
    add_definitions(-DAlphaOnePerfect)
ENDIF()

message("Coverage flag = ${COVERAGE}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-std=c++2a ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g")

set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Werror=deprecated-enum-float-conversion -Wno-error=strict-overflow -Wno-error=strict-aliasing -O3")

if(CHECK_MEMORY)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -static-libasan")
    message("Check Memory CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
endif()

if(PERF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
endif()

if(COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    message("Coverage CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS_EXTRA "-pedantic -Wall -Wextra -Wcast-align -Werror=deprecated-enum-float-conversion -Wno-coverage-mismatch -Wnull-dereference -Wno-return-local-addr -Wunused-const-variable=1 -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Woverloaded-virtual -Wredundant-decls -Wshadow -Wstrict-null-sentinel -Wswitch-default -Wundef -Wno-unused -Wextra-semi -Werror")

include_directories(
    "/usr/local/include"
    "/usr/include"
    ${CURL_INCLUDE_DIRS}
)

link_directories(
    "/usr/local/lib"
    "/usr/lib"
    "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}"
    ${CURL_LIBRARIES}
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../deploy)

add_subdirectory(src)

# Find the AWS SDK for C++ package.
find_package(AWSSDK REQUIRED COMPONENTS s3)
add_executable("s3" s3.cpp)
target_link_libraries("s3" ${AWSSDK_LINK_LIBRARIES})