# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build

# Include any dependencies generated for this target.
include src/colmap/exe/CMakeFiles/colmap_main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/colmap/exe/CMakeFiles/colmap_main.dir/compiler_depend.make

# Include the progress variables for this target.
include src/colmap/exe/CMakeFiles/colmap_main.dir/progress.make

# Include the compile flags for this target's objects.
include src/colmap/exe/CMakeFiles/colmap_main.dir/flags.make

src/colmap/exe/CMakeFiles/colmap_main.dir/feature.cc.o: src/colmap/exe/CMakeFiles/colmap_main.dir/flags.make
src/colmap/exe/CMakeFiles/colmap_main.dir/feature.cc.o: /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/feature.cc
src/colmap/exe/CMakeFiles/colmap_main.dir/feature.cc.o: src/colmap/exe/CMakeFiles/colmap_main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/colmap/exe/CMakeFiles/colmap_main.dir/feature.cc.o"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/colmap/exe/CMakeFiles/colmap_main.dir/feature.cc.o -MF CMakeFiles/colmap_main.dir/feature.cc.o.d -o CMakeFiles/colmap_main.dir/feature.cc.o -c /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/feature.cc

src/colmap/exe/CMakeFiles/colmap_main.dir/feature.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/colmap_main.dir/feature.cc.i"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/feature.cc > CMakeFiles/colmap_main.dir/feature.cc.i

src/colmap/exe/CMakeFiles/colmap_main.dir/feature.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/colmap_main.dir/feature.cc.s"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/feature.cc -o CMakeFiles/colmap_main.dir/feature.cc.s

src/colmap/exe/CMakeFiles/colmap_main.dir/sfm.cc.o: src/colmap/exe/CMakeFiles/colmap_main.dir/flags.make
src/colmap/exe/CMakeFiles/colmap_main.dir/sfm.cc.o: /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/sfm.cc
src/colmap/exe/CMakeFiles/colmap_main.dir/sfm.cc.o: src/colmap/exe/CMakeFiles/colmap_main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/colmap/exe/CMakeFiles/colmap_main.dir/sfm.cc.o"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/colmap/exe/CMakeFiles/colmap_main.dir/sfm.cc.o -MF CMakeFiles/colmap_main.dir/sfm.cc.o.d -o CMakeFiles/colmap_main.dir/sfm.cc.o -c /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/sfm.cc

src/colmap/exe/CMakeFiles/colmap_main.dir/sfm.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/colmap_main.dir/sfm.cc.i"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/sfm.cc > CMakeFiles/colmap_main.dir/sfm.cc.i

src/colmap/exe/CMakeFiles/colmap_main.dir/sfm.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/colmap_main.dir/sfm.cc.s"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/sfm.cc -o CMakeFiles/colmap_main.dir/sfm.cc.s

src/colmap/exe/CMakeFiles/colmap_main.dir/colmap.cc.o: src/colmap/exe/CMakeFiles/colmap_main.dir/flags.make
src/colmap/exe/CMakeFiles/colmap_main.dir/colmap.cc.o: /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/colmap.cc
src/colmap/exe/CMakeFiles/colmap_main.dir/colmap.cc.o: src/colmap/exe/CMakeFiles/colmap_main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/colmap/exe/CMakeFiles/colmap_main.dir/colmap.cc.o"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/colmap/exe/CMakeFiles/colmap_main.dir/colmap.cc.o -MF CMakeFiles/colmap_main.dir/colmap.cc.o.d -o CMakeFiles/colmap_main.dir/colmap.cc.o -c /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/colmap.cc

src/colmap/exe/CMakeFiles/colmap_main.dir/colmap.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/colmap_main.dir/colmap.cc.i"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/colmap.cc > CMakeFiles/colmap_main.dir/colmap.cc.i

src/colmap/exe/CMakeFiles/colmap_main.dir/colmap.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/colmap_main.dir/colmap.cc.s"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/colmap.cc -o CMakeFiles/colmap_main.dir/colmap.cc.s

src/colmap/exe/CMakeFiles/colmap_main.dir/database.cc.o: src/colmap/exe/CMakeFiles/colmap_main.dir/flags.make
src/colmap/exe/CMakeFiles/colmap_main.dir/database.cc.o: /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/database.cc
src/colmap/exe/CMakeFiles/colmap_main.dir/database.cc.o: src/colmap/exe/CMakeFiles/colmap_main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/colmap/exe/CMakeFiles/colmap_main.dir/database.cc.o"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/colmap/exe/CMakeFiles/colmap_main.dir/database.cc.o -MF CMakeFiles/colmap_main.dir/database.cc.o.d -o CMakeFiles/colmap_main.dir/database.cc.o -c /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/database.cc

src/colmap/exe/CMakeFiles/colmap_main.dir/database.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/colmap_main.dir/database.cc.i"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/database.cc > CMakeFiles/colmap_main.dir/database.cc.i

src/colmap/exe/CMakeFiles/colmap_main.dir/database.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/colmap_main.dir/database.cc.s"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/database.cc -o CMakeFiles/colmap_main.dir/database.cc.s

src/colmap/exe/CMakeFiles/colmap_main.dir/gui.cc.o: src/colmap/exe/CMakeFiles/colmap_main.dir/flags.make
src/colmap/exe/CMakeFiles/colmap_main.dir/gui.cc.o: /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/gui.cc
src/colmap/exe/CMakeFiles/colmap_main.dir/gui.cc.o: src/colmap/exe/CMakeFiles/colmap_main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/colmap/exe/CMakeFiles/colmap_main.dir/gui.cc.o"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/colmap/exe/CMakeFiles/colmap_main.dir/gui.cc.o -MF CMakeFiles/colmap_main.dir/gui.cc.o.d -o CMakeFiles/colmap_main.dir/gui.cc.o -c /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/gui.cc

src/colmap/exe/CMakeFiles/colmap_main.dir/gui.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/colmap_main.dir/gui.cc.i"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/gui.cc > CMakeFiles/colmap_main.dir/gui.cc.i

src/colmap/exe/CMakeFiles/colmap_main.dir/gui.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/colmap_main.dir/gui.cc.s"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/gui.cc -o CMakeFiles/colmap_main.dir/gui.cc.s

src/colmap/exe/CMakeFiles/colmap_main.dir/image.cc.o: src/colmap/exe/CMakeFiles/colmap_main.dir/flags.make
src/colmap/exe/CMakeFiles/colmap_main.dir/image.cc.o: /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/image.cc
src/colmap/exe/CMakeFiles/colmap_main.dir/image.cc.o: src/colmap/exe/CMakeFiles/colmap_main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/colmap/exe/CMakeFiles/colmap_main.dir/image.cc.o"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/colmap/exe/CMakeFiles/colmap_main.dir/image.cc.o -MF CMakeFiles/colmap_main.dir/image.cc.o.d -o CMakeFiles/colmap_main.dir/image.cc.o -c /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/image.cc

src/colmap/exe/CMakeFiles/colmap_main.dir/image.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/colmap_main.dir/image.cc.i"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/image.cc > CMakeFiles/colmap_main.dir/image.cc.i

src/colmap/exe/CMakeFiles/colmap_main.dir/image.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/colmap_main.dir/image.cc.s"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/image.cc -o CMakeFiles/colmap_main.dir/image.cc.s

src/colmap/exe/CMakeFiles/colmap_main.dir/model.cc.o: src/colmap/exe/CMakeFiles/colmap_main.dir/flags.make
src/colmap/exe/CMakeFiles/colmap_main.dir/model.cc.o: /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/model.cc
src/colmap/exe/CMakeFiles/colmap_main.dir/model.cc.o: src/colmap/exe/CMakeFiles/colmap_main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object src/colmap/exe/CMakeFiles/colmap_main.dir/model.cc.o"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/colmap/exe/CMakeFiles/colmap_main.dir/model.cc.o -MF CMakeFiles/colmap_main.dir/model.cc.o.d -o CMakeFiles/colmap_main.dir/model.cc.o -c /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/model.cc

src/colmap/exe/CMakeFiles/colmap_main.dir/model.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/colmap_main.dir/model.cc.i"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/model.cc > CMakeFiles/colmap_main.dir/model.cc.i

src/colmap/exe/CMakeFiles/colmap_main.dir/model.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/colmap_main.dir/model.cc.s"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/model.cc -o CMakeFiles/colmap_main.dir/model.cc.s

src/colmap/exe/CMakeFiles/colmap_main.dir/mvs.cc.o: src/colmap/exe/CMakeFiles/colmap_main.dir/flags.make
src/colmap/exe/CMakeFiles/colmap_main.dir/mvs.cc.o: /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/mvs.cc
src/colmap/exe/CMakeFiles/colmap_main.dir/mvs.cc.o: src/colmap/exe/CMakeFiles/colmap_main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object src/colmap/exe/CMakeFiles/colmap_main.dir/mvs.cc.o"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/colmap/exe/CMakeFiles/colmap_main.dir/mvs.cc.o -MF CMakeFiles/colmap_main.dir/mvs.cc.o.d -o CMakeFiles/colmap_main.dir/mvs.cc.o -c /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/mvs.cc

src/colmap/exe/CMakeFiles/colmap_main.dir/mvs.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/colmap_main.dir/mvs.cc.i"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/mvs.cc > CMakeFiles/colmap_main.dir/mvs.cc.i

src/colmap/exe/CMakeFiles/colmap_main.dir/mvs.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/colmap_main.dir/mvs.cc.s"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/mvs.cc -o CMakeFiles/colmap_main.dir/mvs.cc.s

src/colmap/exe/CMakeFiles/colmap_main.dir/vocab_tree.cc.o: src/colmap/exe/CMakeFiles/colmap_main.dir/flags.make
src/colmap/exe/CMakeFiles/colmap_main.dir/vocab_tree.cc.o: /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/vocab_tree.cc
src/colmap/exe/CMakeFiles/colmap_main.dir/vocab_tree.cc.o: src/colmap/exe/CMakeFiles/colmap_main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object src/colmap/exe/CMakeFiles/colmap_main.dir/vocab_tree.cc.o"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/colmap/exe/CMakeFiles/colmap_main.dir/vocab_tree.cc.o -MF CMakeFiles/colmap_main.dir/vocab_tree.cc.o.d -o CMakeFiles/colmap_main.dir/vocab_tree.cc.o -c /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/vocab_tree.cc

src/colmap/exe/CMakeFiles/colmap_main.dir/vocab_tree.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/colmap_main.dir/vocab_tree.cc.i"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/vocab_tree.cc > CMakeFiles/colmap_main.dir/vocab_tree.cc.i

src/colmap/exe/CMakeFiles/colmap_main.dir/vocab_tree.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/colmap_main.dir/vocab_tree.cc.s"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe/vocab_tree.cc -o CMakeFiles/colmap_main.dir/vocab_tree.cc.s

# Object files for target colmap_main
colmap_main_OBJECTS = \
"CMakeFiles/colmap_main.dir/feature.cc.o" \
"CMakeFiles/colmap_main.dir/sfm.cc.o" \
"CMakeFiles/colmap_main.dir/colmap.cc.o" \
"CMakeFiles/colmap_main.dir/database.cc.o" \
"CMakeFiles/colmap_main.dir/gui.cc.o" \
"CMakeFiles/colmap_main.dir/image.cc.o" \
"CMakeFiles/colmap_main.dir/model.cc.o" \
"CMakeFiles/colmap_main.dir/mvs.cc.o" \
"CMakeFiles/colmap_main.dir/vocab_tree.cc.o"

# External object files for target colmap_main
colmap_main_EXTERNAL_OBJECTS =

src/colmap/exe/colmap: src/colmap/exe/CMakeFiles/colmap_main.dir/feature.cc.o
src/colmap/exe/colmap: src/colmap/exe/CMakeFiles/colmap_main.dir/sfm.cc.o
src/colmap/exe/colmap: src/colmap/exe/CMakeFiles/colmap_main.dir/colmap.cc.o
src/colmap/exe/colmap: src/colmap/exe/CMakeFiles/colmap_main.dir/database.cc.o
src/colmap/exe/colmap: src/colmap/exe/CMakeFiles/colmap_main.dir/gui.cc.o
src/colmap/exe/colmap: src/colmap/exe/CMakeFiles/colmap_main.dir/image.cc.o
src/colmap/exe/colmap: src/colmap/exe/CMakeFiles/colmap_main.dir/model.cc.o
src/colmap/exe/colmap: src/colmap/exe/CMakeFiles/colmap_main.dir/mvs.cc.o
src/colmap/exe/colmap: src/colmap/exe/CMakeFiles/colmap_main.dir/vocab_tree.cc.o
src/colmap/exe/colmap: src/colmap/exe/CMakeFiles/colmap_main.dir/build.make
src/colmap/exe/colmap: src/colmap/controllers/libcolmap_controllers.a
src/colmap/exe/colmap: src/colmap/retrieval/libcolmap_retrieval.a
src/colmap/exe/colmap: src/colmap/scene/libcolmap_scene.a
src/colmap/exe/colmap: src/colmap/sfm/libcolmap_sfm.a
src/colmap/exe/colmap: src/colmap/util/libcolmap_util.a
src/colmap/exe/colmap: src/colmap/util/libcolmap_util_cuda.a
src/colmap/exe/colmap: src/colmap/mvs/libcolmap_mvs_cuda.a
src/colmap/exe/colmap: src/colmap/ui/libcolmap_ui.a
src/colmap/exe/colmap: src/colmap/controllers/libcolmap_controllers.a
src/colmap/exe/colmap: src/colmap/sfm/libcolmap_sfm.a
src/colmap/exe/colmap: src/colmap/mvs/libcolmap_mvs.a
src/colmap/exe/colmap: src/thirdparty/PoissonRecon/libcolmap_poisson_recon.a
src/colmap/exe/colmap: /home/ghryu/.conda/envs/gh-3dgs/lib/libmpfr.so
src/colmap/exe/colmap: /home/ghryu/.conda/envs/gh-3dgs/lib/libgmp.so
src/colmap/exe/colmap: src/colmap/feature/libcolmap_feature.a
src/colmap/exe/colmap: src/colmap/retrieval/libcolmap_retrieval.a
src/colmap/exe/colmap: src/colmap/estimators/libcolmap_estimators.a
src/colmap/exe/colmap: src/colmap/util/libcolmap_util_cuda.a
src/colmap/exe/colmap: _deps/poselib-build/PoseLib/libPoseLib.a
src/colmap/exe/colmap: src/colmap/optim/libcolmap_optim.a
src/colmap/exe/colmap: /home/ghryu/.conda/envs/gh-3dgs/lib/libflann.so
src/colmap/exe/colmap: /home/ghryu/.conda/envs/gh-3dgs/lib/liblz4.so
src/colmap/exe/colmap: src/thirdparty/SiftGPU/libcolmap_sift_gpu.a
src/colmap/exe/colmap: /home/ghryu/.conda/envs/gh-3dgs/lib/libcudart.so
src/colmap/exe/colmap: /home/ghryu/.conda/envs/gh-3dgs/lib/libcurand.so
src/colmap/exe/colmap: /usr/lib/x86_64-linux-gnu/libGLEW.so
src/colmap/exe/colmap: /usr/lib/x86_64-linux-gnu/libboost_program_options.so.1.71.0
src/colmap/exe/colmap: src/colmap/image/libcolmap_image.a
src/colmap/exe/colmap: src/colmap/scene/libcolmap_scene.a
src/colmap/exe/colmap: src/colmap/geometry/libcolmap_geometry.a
src/colmap/exe/colmap: src/colmap/math/libcolmap_math.a
src/colmap/exe/colmap: /usr/lib/x86_64-linux-gnu/libmetis.so
src/colmap/exe/colmap: /usr/lib/x86_64-linux-gnu/libboost_graph.so.1.71.0
src/colmap/exe/colmap: /usr/lib/x86_64-linux-gnu/libboost_regex.so.1.71.0
src/colmap/exe/colmap: src/colmap/feature/libcolmap_feature_types.a
src/colmap/exe/colmap: src/colmap/sensor/libcolmap_sensor.a
src/colmap/exe/colmap: src/colmap/util/libcolmap_util.a
src/colmap/exe/colmap: /home/ghryu/.conda/envs/gh-3dgs/lib/libsqlite3.so
src/colmap/exe/colmap: /home/ghryu/.conda/envs/gh-3dgs/lib/libcurl.so
src/colmap/exe/colmap: /home/ghryu/.conda/envs/gh-3dgs/lib/libcrypto.so
src/colmap/exe/colmap: /usr/lib/x86_64-linux-gnu/libGLX.so
src/colmap/exe/colmap: /usr/lib/x86_64-linux-gnu/libOpenGL.so
src/colmap/exe/colmap: /home/ghryu/.conda/envs/gh-3dgs/lib/libceres.so.2.2.0
src/colmap/exe/colmap: /home/ghryu/.conda/envs/gh-3dgs/lib/libglog.so.0.7.1
src/colmap/exe/colmap: /home/ghryu/.conda/envs/gh-3dgs/lib/libgflags.so.2.2.2
src/colmap/exe/colmap: src/thirdparty/VLFeat/libcolmap_vlfeat.a
src/colmap/exe/colmap: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
src/colmap/exe/colmap: /usr/lib/x86_64-linux-gnu/libpthread.so
src/colmap/exe/colmap: /usr/lib/x86_64-linux-gnu/libfreeimage.so
src/colmap/exe/colmap: src/thirdparty/LSD/libcolmap_lsd.a
src/colmap/exe/colmap: /usr/lib/x86_64-linux-gnu/libQt5OpenGL.so.5.12.8
src/colmap/exe/colmap: /usr/lib/x86_64-linux-gnu/libQt5Widgets.so.5.12.8
src/colmap/exe/colmap: /usr/lib/x86_64-linux-gnu/libQt5Gui.so.5.12.8
src/colmap/exe/colmap: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5.12.8
src/colmap/exe/colmap: src/colmap/exe/CMakeFiles/colmap_main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX executable colmap"
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/colmap_main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/colmap/exe/CMakeFiles/colmap_main.dir/build: src/colmap/exe/colmap
.PHONY : src/colmap/exe/CMakeFiles/colmap_main.dir/build

src/colmap/exe/CMakeFiles/colmap_main.dir/clean:
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe && $(CMAKE_COMMAND) -P CMakeFiles/colmap_main.dir/cmake_clean.cmake
.PHONY : src/colmap/exe/CMakeFiles/colmap_main.dir/clean

src/colmap/exe/CMakeFiles/colmap_main.dir/depend:
	cd /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/src/colmap/exe /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe /home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/src/colmap/exe/CMakeFiles/colmap_main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/colmap/exe/CMakeFiles/colmap_main.dir/depend

