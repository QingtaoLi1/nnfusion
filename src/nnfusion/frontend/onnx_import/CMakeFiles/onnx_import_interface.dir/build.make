# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

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
CMAKE_COMMAND = /opt/conda/bin/cmake

# The command to remove a file.
RM = /opt/conda/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /workspace/v-leiwang3/nnfusion_xbox

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspace/v-leiwang3/nnfusion_xbox

# Include any dependencies generated for this target.
include src/nnfusion/frontend/onnx_import/CMakeFiles/onnx_import_interface.dir/depend.make

# Include the progress variables for this target.
include src/nnfusion/frontend/onnx_import/CMakeFiles/onnx_import_interface.dir/progress.make

# Include the compile flags for this target's objects.
include src/nnfusion/frontend/onnx_import/CMakeFiles/onnx_import_interface.dir/flags.make

src/nnfusion/frontend/onnx_import/CMakeFiles/onnx_import_interface.dir/onnx.cpp.o: src/nnfusion/frontend/onnx_import/CMakeFiles/onnx_import_interface.dir/flags.make
src/nnfusion/frontend/onnx_import/CMakeFiles/onnx_import_interface.dir/onnx.cpp.o: src/nnfusion/frontend/onnx_import/onnx.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/nnfusion/frontend/onnx_import/CMakeFiles/onnx_import_interface.dir/onnx.cpp.o"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/frontend/onnx_import && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/onnx_import_interface.dir/onnx.cpp.o -c /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/frontend/onnx_import/onnx.cpp

src/nnfusion/frontend/onnx_import/CMakeFiles/onnx_import_interface.dir/onnx.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/onnx_import_interface.dir/onnx.cpp.i"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/frontend/onnx_import && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/frontend/onnx_import/onnx.cpp > CMakeFiles/onnx_import_interface.dir/onnx.cpp.i

src/nnfusion/frontend/onnx_import/CMakeFiles/onnx_import_interface.dir/onnx.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/onnx_import_interface.dir/onnx.cpp.s"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/frontend/onnx_import && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/frontend/onnx_import/onnx.cpp -o CMakeFiles/onnx_import_interface.dir/onnx.cpp.s

# Object files for target onnx_import_interface
onnx_import_interface_OBJECTS = \
"CMakeFiles/onnx_import_interface.dir/onnx.cpp.o"

# External object files for target onnx_import_interface
onnx_import_interface_EXTERNAL_OBJECTS =

src/nnfusion/frontend/onnx_import/libonnx_import_interface.a: src/nnfusion/frontend/onnx_import/CMakeFiles/onnx_import_interface.dir/onnx.cpp.o
src/nnfusion/frontend/onnx_import/libonnx_import_interface.a: src/nnfusion/frontend/onnx_import/CMakeFiles/onnx_import_interface.dir/build.make
src/nnfusion/frontend/onnx_import/libonnx_import_interface.a: src/nnfusion/frontend/onnx_import/CMakeFiles/onnx_import_interface.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libonnx_import_interface.a"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/frontend/onnx_import && $(CMAKE_COMMAND) -P CMakeFiles/onnx_import_interface.dir/cmake_clean_target.cmake
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/frontend/onnx_import && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/onnx_import_interface.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/nnfusion/frontend/onnx_import/CMakeFiles/onnx_import_interface.dir/build: src/nnfusion/frontend/onnx_import/libonnx_import_interface.a

.PHONY : src/nnfusion/frontend/onnx_import/CMakeFiles/onnx_import_interface.dir/build

src/nnfusion/frontend/onnx_import/CMakeFiles/onnx_import_interface.dir/clean:
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/frontend/onnx_import && $(CMAKE_COMMAND) -P CMakeFiles/onnx_import_interface.dir/cmake_clean.cmake
.PHONY : src/nnfusion/frontend/onnx_import/CMakeFiles/onnx_import_interface.dir/clean

src/nnfusion/frontend/onnx_import/CMakeFiles/onnx_import_interface.dir/depend:
	cd /workspace/v-leiwang3/nnfusion_xbox && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/v-leiwang3/nnfusion_xbox /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/frontend/onnx_import /workspace/v-leiwang3/nnfusion_xbox /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/frontend/onnx_import /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/frontend/onnx_import/CMakeFiles/onnx_import_interface.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/nnfusion/frontend/onnx_import/CMakeFiles/onnx_import_interface.dir/depend

