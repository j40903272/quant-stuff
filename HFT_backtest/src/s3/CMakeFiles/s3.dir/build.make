# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/s3

# Include any dependencies generated for this target.
include CMakeFiles/s3.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/s3.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/s3.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/s3.dir/flags.make

CMakeFiles/s3.dir/s3.cpp.o: CMakeFiles/s3.dir/flags.make
CMakeFiles/s3.dir/s3.cpp.o: ../s3.cpp
CMakeFiles/s3.dir/s3.cpp.o: CMakeFiles/s3.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/s3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/s3.dir/s3.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/s3.dir/s3.cpp.o -MF CMakeFiles/s3.dir/s3.cpp.o.d -o CMakeFiles/s3.dir/s3.cpp.o -c /home/ubuntu/s3.cpp

CMakeFiles/s3.dir/s3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/s3.dir/s3.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/s3.cpp > CMakeFiles/s3.dir/s3.cpp.i

CMakeFiles/s3.dir/s3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/s3.dir/s3.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/s3.cpp -o CMakeFiles/s3.dir/s3.cpp.s

# Object files for target s3
s3_OBJECTS = \
"CMakeFiles/s3.dir/s3.cpp.o"

# External object files for target s3
s3_EXTERNAL_OBJECTS =

s3: CMakeFiles/s3.dir/s3.cpp.o
s3: CMakeFiles/s3.dir/build.make
s3: /usr/local/lib/libaws-cpp-sdk-s3.so
s3: /usr/local/lib/libaws-cpp-sdk-core.so
s3: /usr/local/lib/libaws-crt-cpp.a
s3: /usr/local/lib/libaws-c-mqtt.a
s3: /usr/local/lib/libaws-c-event-stream.a
s3: /usr/local/lib/libaws-c-s3.a
s3: /usr/local/lib/libaws-c-auth.a
s3: /usr/local/lib/libaws-c-http.a
s3: /usr/local/lib/libaws-c-io.a
s3: /usr/local/lib/libs2n.a
s3: /usr/lib/x86_64-linux-gnu/libcrypto.a
s3: /usr/local/lib/libaws-c-compression.a
s3: /usr/local/lib/libaws-c-cal.a
s3: /usr/lib/x86_64-linux-gnu/libcrypto.so
s3: /usr/local/lib/libaws-c-sdkutils.a
s3: /usr/local/lib/libaws-checksums.a
s3: /usr/local/lib/libaws-c-common.a
s3: CMakeFiles/s3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/s3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable s3"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/s3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/s3.dir/build: s3
.PHONY : CMakeFiles/s3.dir/build

CMakeFiles/s3.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/s3.dir/cmake_clean.cmake
.PHONY : CMakeFiles/s3.dir/clean

CMakeFiles/s3.dir/depend:
	cd /home/ubuntu/s3 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu /home/ubuntu /home/ubuntu/s3 /home/ubuntu/s3 /home/ubuntu/s3/CMakeFiles/s3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/s3.dir/depend
