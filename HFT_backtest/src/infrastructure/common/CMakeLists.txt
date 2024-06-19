# Create a library called "alphaone_common".
# The extension is already found. Any number of sources could be listed here.

set (TARGET_NAME "alphaone_common")

file (GLOB MAIN_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*/*/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*/*/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*/*/*.c
)

file (GLOB MAIN_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*/*/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*/*/*/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*/*/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*/*/*/*.hpp
)

file (GLOB GTEST_DISABLE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/gtest/*.cc
)

file (GLOB GMOCK_DISABLE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/gmock/*.cc
)

file (GLOB BENCHMARK_DISABLE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/src/*.cc
)

file (GLOB PROTOBUF_DISABLE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/*.cpp
)

file (GLOB SPDLOG_DISABLE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/spdlog/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/spdlog/*/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/spdlog/*/*/*.h
)

add_library (${TARGET_NAME} STATIC
    ${MAIN_SRC}
    ${MAIN_HEADERS}
)

target_include_directories (${TARGET_NAME} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../.."
)

target_link_libraries (${TARGET_NAME}
    z
)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_EXTRA} -DBOOST_SYSTEM_NO_DEPRECATED")

# DISABLE SOME EXTERN PROJECT WITH CERTAIN COMPILE OPTIONS
if (CMAKE_BUILD_TYPE MATCHES Debug)
    message("GTEST_DISABLE_SRC = ${GTEST_DISABLE_SRC}")
    message("GMOCK_DISABLE_SRC = ${GMOCK_DISABLE_SRC}")
    message("GOOGLE_BENCHMARK_DISABLE_SRC = ${BENCHMARK_DISABLE_SRC}")
    message("PROTOBUF_DISABLE_SRC = ${PROTOBUF_DISABLE_SRC}")
    message("SPDLOG_DISABLE_HEADERS = ${SPDLOG_DISABLE_HEADERS}")
endif()

set_source_files_properties(${GTEST_DISABLE_SRC} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wno-undef -Wno-ctor-dtor-privacy -Wno-switch-default -Wno-missing-declarations")
set_source_files_properties(${GMOCK_DISABLE_SRC} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wno-undef -Wno-ctor-dtor-privacy -Wno-switch-default -Wno-missing-declarations")
set_source_files_properties(${BENCHMARK_DISABLE_SRC} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wno-noexcept -Wno-redundant-decls -Wno-undef -Wno-ctor-dtor-privacy -Wno-switch-default -Wno-missing-declarations")
set_source_files_properties(${PROTOBUF_DISABLE_SRC} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch-default")
# set_source_files_properties(${SPDLOG_DISABLE_HEADERS} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ctor-dtor-privacy")
