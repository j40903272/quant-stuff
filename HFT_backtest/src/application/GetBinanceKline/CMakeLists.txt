set (TARGET_NAME "GetBinanceKline")

file (GLOB MAIN_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file (GLOB MAIN_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

add_executable (${TARGET_NAME}
    ${MAIN_SRC}
    ${MAIN_HEADERS}
)

target_include_directories (${TARGET_NAME} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

set (CMAKE_CXX_FLAGS "-g -DBOOST_SYSTEM_NO_DEPRECATED")
set (CMAKE_CXX_FLAGS "-std=c++2a ${CMAKE_CXX_FLAGS}")

target_link_libraries (${TARGET_NAME}
    curl
    boost_program_options
    protobuf
    pthread
    z
    alphaone_status
    alphaone_platform
    alphaone_common
    alphaone_counter
    alphaone_strategy
)