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
include src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/depend.make

# Include the progress variables for this target.
include src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/progress.make

# Include the compile flags for this target's objects.
include src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/flags.make

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/extract_graph_signature.cpp.o: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/flags.make
src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/extract_graph_signature.cpp.o: src/nnfusion/engine/pass/extract_graph_signature.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/extract_graph_signature.cpp.o"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nnfusion_engine_pass.dir/extract_graph_signature.cpp.o -c /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/extract_graph_signature.cpp

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/extract_graph_signature.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nnfusion_engine_pass.dir/extract_graph_signature.cpp.i"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/extract_graph_signature.cpp > CMakeFiles/nnfusion_engine_pass.dir/extract_graph_signature.cpp.i

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/extract_graph_signature.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nnfusion_engine_pass.dir/extract_graph_signature.cpp.s"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/extract_graph_signature.cpp -o CMakeFiles/nnfusion_engine_pass.dir/extract_graph_signature.cpp.s

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/base_codegen_pass.cpp.o: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/flags.make
src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/base_codegen_pass.cpp.o: src/nnfusion/engine/pass/codegen/base_codegen_pass.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/base_codegen_pass.cpp.o"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nnfusion_engine_pass.dir/codegen/base_codegen_pass.cpp.o -c /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/base_codegen_pass.cpp

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/base_codegen_pass.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nnfusion_engine_pass.dir/codegen/base_codegen_pass.cpp.i"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/base_codegen_pass.cpp > CMakeFiles/nnfusion_engine_pass.dir/codegen/base_codegen_pass.cpp.i

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/base_codegen_pass.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nnfusion_engine_pass.dir/codegen/base_codegen_pass.cpp.s"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/base_codegen_pass.cpp -o CMakeFiles/nnfusion_engine_pass.dir/codegen/base_codegen_pass.cpp.s

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/codegen_langunit.cpp.o: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/flags.make
src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/codegen_langunit.cpp.o: src/nnfusion/engine/pass/codegen/codegen_langunit.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/codegen_langunit.cpp.o"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nnfusion_engine_pass.dir/codegen/codegen_langunit.cpp.o -c /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/codegen_langunit.cpp

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/codegen_langunit.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nnfusion_engine_pass.dir/codegen/codegen_langunit.cpp.i"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/codegen_langunit.cpp > CMakeFiles/nnfusion_engine_pass.dir/codegen/codegen_langunit.cpp.i

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/codegen_langunit.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nnfusion_engine_pass.dir/codegen/codegen_langunit.cpp.s"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/codegen_langunit.cpp -o CMakeFiles/nnfusion_engine_pass.dir/codegen/codegen_langunit.cpp.s

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator.cpp.o: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/flags.make
src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator.cpp.o: src/nnfusion/engine/pass/codegen/codegenerator.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator.cpp.o"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator.cpp.o -c /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/codegenerator.cpp

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator.cpp.i"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/codegenerator.cpp > CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator.cpp.i

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator.cpp.s"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/codegenerator.cpp -o CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator.cpp.s

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator_helper.cpp.o: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/flags.make
src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator_helper.cpp.o: src/nnfusion/engine/pass/codegen/codegenerator_helper.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator_helper.cpp.o"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator_helper.cpp.o -c /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/codegenerator_helper.cpp

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator_helper.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator_helper.cpp.i"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/codegenerator_helper.cpp > CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator_helper.cpp.i

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator_helper.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator_helper.cpp.s"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/codegenerator_helper.cpp -o CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator_helper.cpp.s

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/cpu_codegen_pass.cpp.o: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/flags.make
src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/cpu_codegen_pass.cpp.o: src/nnfusion/engine/pass/codegen/cpu_codegen_pass.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/cpu_codegen_pass.cpp.o"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nnfusion_engine_pass.dir/codegen/cpu_codegen_pass.cpp.o -c /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/cpu_codegen_pass.cpp

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/cpu_codegen_pass.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nnfusion_engine_pass.dir/codegen/cpu_codegen_pass.cpp.i"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/cpu_codegen_pass.cpp > CMakeFiles/nnfusion_engine_pass.dir/codegen/cpu_codegen_pass.cpp.i

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/cpu_codegen_pass.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nnfusion_engine_pass.dir/codegen/cpu_codegen_pass.cpp.s"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/cpu_codegen_pass.cpp -o CMakeFiles/nnfusion_engine_pass.dir/codegen/cpu_codegen_pass.cpp.s

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/cuda_codegen_pass.cpp.o: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/flags.make
src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/cuda_codegen_pass.cpp.o: src/nnfusion/engine/pass/codegen/cuda_codegen_pass.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/cuda_codegen_pass.cpp.o"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nnfusion_engine_pass.dir/codegen/cuda_codegen_pass.cpp.o -c /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/cuda_codegen_pass.cpp

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/cuda_codegen_pass.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nnfusion_engine_pass.dir/codegen/cuda_codegen_pass.cpp.i"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/cuda_codegen_pass.cpp > CMakeFiles/nnfusion_engine_pass.dir/codegen/cuda_codegen_pass.cpp.i

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/cuda_codegen_pass.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nnfusion_engine_pass.dir/codegen/cuda_codegen_pass.cpp.s"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/cuda_codegen_pass.cpp -o CMakeFiles/nnfusion_engine_pass.dir/codegen/cuda_codegen_pass.cpp.s

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_codegen_pass.cpp.o: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/flags.make
src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_codegen_pass.cpp.o: src/nnfusion/engine/pass/codegen/hlsl_codegen_pass.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_codegen_pass.cpp.o"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_codegen_pass.cpp.o -c /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/hlsl_codegen_pass.cpp

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_codegen_pass.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_codegen_pass.cpp.i"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/hlsl_codegen_pass.cpp > CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_codegen_pass.cpp.i

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_codegen_pass.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_codegen_pass.cpp.s"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/hlsl_codegen_pass.cpp -o CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_codegen_pass.cpp.s

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cpp_codegen_pass.cpp.o: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/flags.make
src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cpp_codegen_pass.cpp.o: src/nnfusion/engine/pass/codegen/hlsl_cpp_codegen_pass.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cpp_codegen_pass.cpp.o"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cpp_codegen_pass.cpp.o -c /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/hlsl_cpp_codegen_pass.cpp

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cpp_codegen_pass.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cpp_codegen_pass.cpp.i"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/hlsl_cpp_codegen_pass.cpp > CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cpp_codegen_pass.cpp.i

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cpp_codegen_pass.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cpp_codegen_pass.cpp.s"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/hlsl_cpp_codegen_pass.cpp -o CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cpp_codegen_pass.cpp.s

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cs_codegen_pass.cpp.o: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/flags.make
src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cs_codegen_pass.cpp.o: src/nnfusion/engine/pass/codegen/hlsl_cs_codegen_pass.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cs_codegen_pass.cpp.o"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cs_codegen_pass.cpp.o -c /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/hlsl_cs_codegen_pass.cpp

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cs_codegen_pass.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cs_codegen_pass.cpp.i"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/hlsl_cs_codegen_pass.cpp > CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cs_codegen_pass.cpp.i

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cs_codegen_pass.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cs_codegen_pass.cpp.s"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/hlsl_cs_codegen_pass.cpp -o CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cs_codegen_pass.cpp.s

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/rocm_codegen_pass.cpp.o: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/flags.make
src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/rocm_codegen_pass.cpp.o: src/nnfusion/engine/pass/codegen/rocm_codegen_pass.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/rocm_codegen_pass.cpp.o"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nnfusion_engine_pass.dir/codegen/rocm_codegen_pass.cpp.o -c /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/rocm_codegen_pass.cpp

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/rocm_codegen_pass.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nnfusion_engine_pass.dir/codegen/rocm_codegen_pass.cpp.i"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/rocm_codegen_pass.cpp > CMakeFiles/nnfusion_engine_pass.dir/codegen/rocm_codegen_pass.cpp.i

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/rocm_codegen_pass.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nnfusion_engine_pass.dir/codegen/rocm_codegen_pass.cpp.s"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/codegen/rocm_codegen_pass.cpp -o CMakeFiles/nnfusion_engine_pass.dir/codegen/rocm_codegen_pass.cpp.s

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/inplace_tensor_analysis.cpp.o: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/flags.make
src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/inplace_tensor_analysis.cpp.o: src/nnfusion/engine/pass/tensor/inplace_tensor_analysis.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/inplace_tensor_analysis.cpp.o"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nnfusion_engine_pass.dir/tensor/inplace_tensor_analysis.cpp.o -c /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/tensor/inplace_tensor_analysis.cpp

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/inplace_tensor_analysis.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nnfusion_engine_pass.dir/tensor/inplace_tensor_analysis.cpp.i"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/tensor/inplace_tensor_analysis.cpp > CMakeFiles/nnfusion_engine_pass.dir/tensor/inplace_tensor_analysis.cpp.i

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/inplace_tensor_analysis.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nnfusion_engine_pass.dir/tensor/inplace_tensor_analysis.cpp.s"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/tensor/inplace_tensor_analysis.cpp -o CMakeFiles/nnfusion_engine_pass.dir/tensor/inplace_tensor_analysis.cpp.s

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/liveness_analysis.cpp.o: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/flags.make
src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/liveness_analysis.cpp.o: src/nnfusion/engine/pass/tensor/liveness_analysis.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/liveness_analysis.cpp.o"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nnfusion_engine_pass.dir/tensor/liveness_analysis.cpp.o -c /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/tensor/liveness_analysis.cpp

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/liveness_analysis.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nnfusion_engine_pass.dir/tensor/liveness_analysis.cpp.i"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/tensor/liveness_analysis.cpp > CMakeFiles/nnfusion_engine_pass.dir/tensor/liveness_analysis.cpp.i

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/liveness_analysis.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nnfusion_engine_pass.dir/tensor/liveness_analysis.cpp.s"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/tensor/liveness_analysis.cpp -o CMakeFiles/nnfusion_engine_pass.dir/tensor/liveness_analysis.cpp.s

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_device_dispatcher.cpp.o: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/flags.make
src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_device_dispatcher.cpp.o: src/nnfusion/engine/pass/tensor/tensor_device_dispatcher.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_device_dispatcher.cpp.o"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_device_dispatcher.cpp.o -c /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/tensor/tensor_device_dispatcher.cpp

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_device_dispatcher.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_device_dispatcher.cpp.i"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/tensor/tensor_device_dispatcher.cpp > CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_device_dispatcher.cpp.i

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_device_dispatcher.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_device_dispatcher.cpp.s"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/tensor/tensor_device_dispatcher.cpp -o CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_device_dispatcher.cpp.s

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_memory_layout.cpp.o: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/flags.make
src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_memory_layout.cpp.o: src/nnfusion/engine/pass/tensor/tensor_memory_layout.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building CXX object src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_memory_layout.cpp.o"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_memory_layout.cpp.o -c /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/tensor/tensor_memory_layout.cpp

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_memory_layout.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_memory_layout.cpp.i"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/tensor/tensor_memory_layout.cpp > CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_memory_layout.cpp.i

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_memory_layout.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_memory_layout.cpp.s"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/tensor/tensor_memory_layout.cpp -o CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_memory_layout.cpp.s

# Object files for target nnfusion_engine_pass
nnfusion_engine_pass_OBJECTS = \
"CMakeFiles/nnfusion_engine_pass.dir/extract_graph_signature.cpp.o" \
"CMakeFiles/nnfusion_engine_pass.dir/codegen/base_codegen_pass.cpp.o" \
"CMakeFiles/nnfusion_engine_pass.dir/codegen/codegen_langunit.cpp.o" \
"CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator.cpp.o" \
"CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator_helper.cpp.o" \
"CMakeFiles/nnfusion_engine_pass.dir/codegen/cpu_codegen_pass.cpp.o" \
"CMakeFiles/nnfusion_engine_pass.dir/codegen/cuda_codegen_pass.cpp.o" \
"CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_codegen_pass.cpp.o" \
"CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cpp_codegen_pass.cpp.o" \
"CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cs_codegen_pass.cpp.o" \
"CMakeFiles/nnfusion_engine_pass.dir/codegen/rocm_codegen_pass.cpp.o" \
"CMakeFiles/nnfusion_engine_pass.dir/tensor/inplace_tensor_analysis.cpp.o" \
"CMakeFiles/nnfusion_engine_pass.dir/tensor/liveness_analysis.cpp.o" \
"CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_device_dispatcher.cpp.o" \
"CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_memory_layout.cpp.o"

# External object files for target nnfusion_engine_pass
nnfusion_engine_pass_EXTERNAL_OBJECTS =

src/nnfusion/engine/pass/libnnfusion_engine_pass.a: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/extract_graph_signature.cpp.o
src/nnfusion/engine/pass/libnnfusion_engine_pass.a: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/base_codegen_pass.cpp.o
src/nnfusion/engine/pass/libnnfusion_engine_pass.a: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/codegen_langunit.cpp.o
src/nnfusion/engine/pass/libnnfusion_engine_pass.a: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator.cpp.o
src/nnfusion/engine/pass/libnnfusion_engine_pass.a: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/codegenerator_helper.cpp.o
src/nnfusion/engine/pass/libnnfusion_engine_pass.a: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/cpu_codegen_pass.cpp.o
src/nnfusion/engine/pass/libnnfusion_engine_pass.a: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/cuda_codegen_pass.cpp.o
src/nnfusion/engine/pass/libnnfusion_engine_pass.a: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_codegen_pass.cpp.o
src/nnfusion/engine/pass/libnnfusion_engine_pass.a: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cpp_codegen_pass.cpp.o
src/nnfusion/engine/pass/libnnfusion_engine_pass.a: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/hlsl_cs_codegen_pass.cpp.o
src/nnfusion/engine/pass/libnnfusion_engine_pass.a: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/codegen/rocm_codegen_pass.cpp.o
src/nnfusion/engine/pass/libnnfusion_engine_pass.a: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/inplace_tensor_analysis.cpp.o
src/nnfusion/engine/pass/libnnfusion_engine_pass.a: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/liveness_analysis.cpp.o
src/nnfusion/engine/pass/libnnfusion_engine_pass.a: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_device_dispatcher.cpp.o
src/nnfusion/engine/pass/libnnfusion_engine_pass.a: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/tensor/tensor_memory_layout.cpp.o
src/nnfusion/engine/pass/libnnfusion_engine_pass.a: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/build.make
src/nnfusion/engine/pass/libnnfusion_engine_pass.a: src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Linking CXX static library libnnfusion_engine_pass.a"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && $(CMAKE_COMMAND) -P CMakeFiles/nnfusion_engine_pass.dir/cmake_clean_target.cmake
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nnfusion_engine_pass.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/build: src/nnfusion/engine/pass/libnnfusion_engine_pass.a

.PHONY : src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/build

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/clean:
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass && $(CMAKE_COMMAND) -P CMakeFiles/nnfusion_engine_pass.dir/cmake_clean.cmake
.PHONY : src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/clean

src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/depend:
	cd /workspace/v-leiwang3/nnfusion_xbox && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/v-leiwang3/nnfusion_xbox /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass /workspace/v-leiwang3/nnfusion_xbox /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/nnfusion/engine/pass/CMakeFiles/nnfusion_engine_pass.dir/depend

