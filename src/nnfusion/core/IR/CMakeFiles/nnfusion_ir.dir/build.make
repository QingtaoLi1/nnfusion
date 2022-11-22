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
include src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/depend.make

# Include the progress variables for this target.
include src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/progress.make

# Include the compile flags for this target's objects.
include src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/flags.make

src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/attribute.cpp.o: src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/flags.make
src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/attribute.cpp.o: src/nnfusion/core/IR/attribute.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/attribute.cpp.o"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nnfusion_ir.dir/attribute.cpp.o -c /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR/attribute.cpp

src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/attribute.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nnfusion_ir.dir/attribute.cpp.i"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR/attribute.cpp > CMakeFiles/nnfusion_ir.dir/attribute.cpp.i

src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/attribute.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nnfusion_ir.dir/attribute.cpp.s"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR/attribute.cpp -o CMakeFiles/nnfusion_ir.dir/attribute.cpp.s

src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/instruction.cpp.o: src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/flags.make
src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/instruction.cpp.o: src/nnfusion/core/IR/instruction.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/instruction.cpp.o"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nnfusion_ir.dir/instruction.cpp.o -c /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR/instruction.cpp

src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/instruction.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nnfusion_ir.dir/instruction.cpp.i"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR/instruction.cpp > CMakeFiles/nnfusion_ir.dir/instruction.cpp.i

src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/instruction.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nnfusion_ir.dir/instruction.cpp.s"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR/instruction.cpp -o CMakeFiles/nnfusion_ir.dir/instruction.cpp.s

src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/program.cpp.o: src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/flags.make
src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/program.cpp.o: src/nnfusion/core/IR/program.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/program.cpp.o"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nnfusion_ir.dir/program.cpp.o -c /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR/program.cpp

src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/program.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nnfusion_ir.dir/program.cpp.i"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR/program.cpp > CMakeFiles/nnfusion_ir.dir/program.cpp.i

src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/program.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nnfusion_ir.dir/program.cpp.s"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR/program.cpp -o CMakeFiles/nnfusion_ir.dir/program.cpp.s

# Object files for target nnfusion_ir
nnfusion_ir_OBJECTS = \
"CMakeFiles/nnfusion_ir.dir/attribute.cpp.o" \
"CMakeFiles/nnfusion_ir.dir/instruction.cpp.o" \
"CMakeFiles/nnfusion_ir.dir/program.cpp.o"

# External object files for target nnfusion_ir
nnfusion_ir_EXTERNAL_OBJECTS =

src/nnfusion/core/IR/libnnfusion_ir.a: src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/attribute.cpp.o
src/nnfusion/core/IR/libnnfusion_ir.a: src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/instruction.cpp.o
src/nnfusion/core/IR/libnnfusion_ir.a: src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/program.cpp.o
src/nnfusion/core/IR/libnnfusion_ir.a: src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/build.make
src/nnfusion/core/IR/libnnfusion_ir.a: src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/v-leiwang3/nnfusion_xbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library libnnfusion_ir.a"
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR && $(CMAKE_COMMAND) -P CMakeFiles/nnfusion_ir.dir/cmake_clean_target.cmake
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nnfusion_ir.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/build: src/nnfusion/core/IR/libnnfusion_ir.a

.PHONY : src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/build

src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/clean:
	cd /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR && $(CMAKE_COMMAND) -P CMakeFiles/nnfusion_ir.dir/cmake_clean.cmake
.PHONY : src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/clean

src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/depend:
	cd /workspace/v-leiwang3/nnfusion_xbox && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/v-leiwang3/nnfusion_xbox /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR /workspace/v-leiwang3/nnfusion_xbox /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR /workspace/v-leiwang3/nnfusion_xbox/src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/nnfusion/core/IR/CMakeFiles/nnfusion_ir.dir/depend

