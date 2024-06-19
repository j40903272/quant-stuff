# Create a library called "alphaone_base".
# The extension is already found. Any number of sources could be listed here.

set (TARGET_NAME "alphaone_base")

file (GLOB MAIN_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file (GLOB MAIN_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

add_library (${TARGET_NAME} STATIC
    ${MAIN_SRC}
    ${MAIN_HEADERS}
)


target_include_directories (${TARGET_NAME} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_EXTRA} -DBOOST_SYSTEM_NO_DEPRECATED")

target_link_libraries (${TARGET_NAME}
    alphaone_status
    alphaone_common
    $<$<BOOL:${COVERAGE} == true>:gcov>
)

add_dependencies (${TARGET_NAME}
    alphaone_status
    alphaone_common
)
