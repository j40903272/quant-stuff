# Create a library called "alphaone_status".
# The extension is already found. Any number of sources could be listed here.

set (TARGET_NAME "alphaone_status")

file (GLOB MAIN_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*/*.cc
)

file (GLOB MAIN_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*/*/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*/*/*.hpp
)

file (GLOB GTEST_DISABLE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/gtest/*.cc
)

file (GLOB GMOCK_DISABLE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/gmock/*.cc
)

file (GLOB PROTOBUF_DISABLE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/*.cpp
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

# DISABLE SOME EXTERN PROJECT WITH CERTAIN COMPILE OPTIONS
message("GTEST_DISABLE_SRC = ${GTEST_DISABLE_SRC}")
message("GMOCK_DISABLE_SRC = ${GMOCK_DISABLE_SRC}")
message("PROTOBUF_DISABLE_SRC = ${PROTOBUF_DISABLE_SRC}")
set_source_files_properties(${GTEST_DISABLE_SRC} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wno-undef -Wno-ctor-dtor-privacy -Wno-switch-default -Wno-missing-declarations")
set_source_files_properties(${GMOCK_DISABLE_SRC} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wno-undef -Wno-ctor-dtor-privacy -Wno-switch-default -Wno-missing-declarations")
set_source_files_properties(${PROTOBUF_DISABLE_SRC} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch-default")

# GET_GIT_COMMIT_HASH
macro(GET_GIT_COMMIT_HASH _git_hash)
    find_package(Git QUIET)
    if(GIT_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} log -1 --pretty=format:%h
            OUTPUT_VARIABLE ${_git_hash}
            OUTPUT_STRIP_TRAILING_WHITESPACE
            WORKING_DIRECTORY
            ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
endmacro()

# GET_GIT_COMMIT_DATE
macro(GET_GIT_COMMIT_DATE _git_date)
    find_package(Git QUIET)
    if(GIT_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} log -1 --format=%ad --date=local
            OUTPUT_VARIABLE ${_git_date}
            OUTPUT_STRIP_TRAILING_WHITESPACE
            WORKING_DIRECTORY
            ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
endmacro()

# GET_GIT_BRANCH
macro(GET_GIT_BRANCH _git_branch)
    find_package(Git QUIET)
    if(GIT_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} symbolic-ref --short -q HEAD
            OUTPUT_VARIABLE ${_git_branch}
            OUTPUT_STRIP_TRAILING_WHITESPACE
            WORKING_DIRECTORY
            ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
endmacro()

set(GIT_BRANCH "")
GET_GIT_BRANCH(GIT_BRANCH)
message(STATUS "Git branch is ${GIT_BRANCH}")
add_definitions("-DGIT_BRANCH=\"${GIT_BRANCH}\"")

set(GIT_COMMIT_HASH "")
GET_GIT_COMMIT_HASH(GIT_COMMIT_HASH)
message(STATUS "Git commit hash is ${GIT_COMMIT_HASH}")
add_definitions("-DGIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\"")

set(GIT_COMMIT_DATE "")
GET_GIT_COMMIT_DATE(GIT_COMMIT_DATE)
message(STATUS "Git commit date is ${GIT_COMMIT_DATE}")
add_definitions("-DGIT_COMMIT_DATE=\"${GIT_COMMIT_DATE}\"")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Version.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/Version.h
)
