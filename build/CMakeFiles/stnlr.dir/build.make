# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jwyang/Researches/lrgan/stnlrbhwd

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jwyang/Researches/lrgan/stnlrbhwd/build

# Include any dependencies generated for this target.
include CMakeFiles/stnlr.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/stnlr.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/stnlr.dir/flags.make

CMakeFiles/stnlr.dir/init.c.o: CMakeFiles/stnlr.dir/flags.make
CMakeFiles/stnlr.dir/init.c.o: ../init.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jwyang/Researches/lrgan/stnlrbhwd/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/stnlr.dir/init.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/stnlr.dir/init.c.o   -c /home/jwyang/Researches/lrgan/stnlrbhwd/init.c

CMakeFiles/stnlr.dir/init.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/stnlr.dir/init.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/jwyang/Researches/lrgan/stnlrbhwd/init.c > CMakeFiles/stnlr.dir/init.c.i

CMakeFiles/stnlr.dir/init.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/stnlr.dir/init.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/jwyang/Researches/lrgan/stnlrbhwd/init.c -o CMakeFiles/stnlr.dir/init.c.s

CMakeFiles/stnlr.dir/init.c.o.requires:
.PHONY : CMakeFiles/stnlr.dir/init.c.o.requires

CMakeFiles/stnlr.dir/init.c.o.provides: CMakeFiles/stnlr.dir/init.c.o.requires
	$(MAKE) -f CMakeFiles/stnlr.dir/build.make CMakeFiles/stnlr.dir/init.c.o.provides.build
.PHONY : CMakeFiles/stnlr.dir/init.c.o.provides

CMakeFiles/stnlr.dir/init.c.o.provides.build: CMakeFiles/stnlr.dir/init.c.o

# Object files for target stnlr
stnlr_OBJECTS = \
"CMakeFiles/stnlr.dir/init.c.o"

# External object files for target stnlr
stnlr_EXTERNAL_OBJECTS =

libstnlr.so: CMakeFiles/stnlr.dir/init.c.o
libstnlr.so: CMakeFiles/stnlr.dir/build.make
libstnlr.so: /home/jwyang/torch/install/lib/libTH.so.0
libstnlr.so: /opt/OpenBLAS/lib/libopenblas.so
libstnlr.so: CMakeFiles/stnlr.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking C shared module libstnlr.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stnlr.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/stnlr.dir/build: libstnlr.so
.PHONY : CMakeFiles/stnlr.dir/build

CMakeFiles/stnlr.dir/requires: CMakeFiles/stnlr.dir/init.c.o.requires
.PHONY : CMakeFiles/stnlr.dir/requires

CMakeFiles/stnlr.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/stnlr.dir/cmake_clean.cmake
.PHONY : CMakeFiles/stnlr.dir/clean

CMakeFiles/stnlr.dir/depend:
	cd /home/jwyang/Researches/lrgan/stnlrbhwd/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jwyang/Researches/lrgan/stnlrbhwd /home/jwyang/Researches/lrgan/stnlrbhwd /home/jwyang/Researches/lrgan/stnlrbhwd/build /home/jwyang/Researches/lrgan/stnlrbhwd/build /home/jwyang/Researches/lrgan/stnlrbhwd/build/CMakeFiles/stnlr.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/stnlr.dir/depend
