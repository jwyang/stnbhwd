# Install script for directory: /home/jwyang/Researches/lrgan/stnlrbhwd

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/home/jwyang/torch/install")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "Release")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "1")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  IF(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnlrbhwd/scm-1/lib/libstnlr.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnlrbhwd/scm-1/lib/libstnlr.so")
    FILE(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnlrbhwd/scm-1/lib/libstnlr.so"
         RPATH "$ORIGIN/../lib:/home/jwyang/torch/install/lib:/opt/OpenBLAS/lib")
  ENDIF()
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnlrbhwd/scm-1/lib" TYPE MODULE FILES "/home/jwyang/Researches/lrgan/stnlrbhwd/build/libstnlr.so")
  IF(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnlrbhwd/scm-1/lib/libstnlr.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnlrbhwd/scm-1/lib/libstnlr.so")
    FILE(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnlrbhwd/scm-1/lib/libstnlr.so"
         OLD_RPATH "/home/jwyang/torch/install/lib:/opt/OpenBLAS/lib:::::::::::::::"
         NEW_RPATH "$ORIGIN/../lib:/home/jwyang/torch/install/lib:/opt/OpenBLAS/lib")
    IF(CMAKE_INSTALL_DO_STRIP)
      EXECUTE_PROCESS(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnlrbhwd/scm-1/lib/libstnlr.so")
    ENDIF(CMAKE_INSTALL_DO_STRIP)
  ENDIF()
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnlrbhwd/scm-1/lua/stnlr" TYPE FILE FILES
    "/home/jwyang/Researches/lrgan/stnlrbhwd/init.lua"
    "/home/jwyang/Researches/lrgan/stnlrbhwd/AffineTransformMatrixGeneratorLR.lua"
    "/home/jwyang/Researches/lrgan/stnlrbhwd/AffineGridGeneratorBHWDLR.lua"
    "/home/jwyang/Researches/lrgan/stnlrbhwd/test.lua"
    "/home/jwyang/Researches/lrgan/stnlrbhwd/BilinearSamplerBHWDLR.lua"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  IF(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnlrbhwd/scm-1/lib/libcustnlr.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnlrbhwd/scm-1/lib/libcustnlr.so")
    FILE(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnlrbhwd/scm-1/lib/libcustnlr.so"
         RPATH "$ORIGIN/../lib:/home/jwyang/torch/install/lib:/usr/local/cuda/lib64:/opt/OpenBLAS/lib")
  ENDIF()
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnlrbhwd/scm-1/lib" TYPE MODULE FILES "/home/jwyang/Researches/lrgan/stnlrbhwd/build/libcustnlr.so")
  IF(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnlrbhwd/scm-1/lib/libcustnlr.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnlrbhwd/scm-1/lib/libcustnlr.so")
    FILE(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnlrbhwd/scm-1/lib/libcustnlr.so"
         OLD_RPATH "/home/jwyang/torch/install/lib:/usr/local/cuda/lib64:/opt/OpenBLAS/lib:::::::::::::::"
         NEW_RPATH "$ORIGIN/../lib:/home/jwyang/torch/install/lib:/usr/local/cuda/lib64:/opt/OpenBLAS/lib")
    IF(CMAKE_INSTALL_DO_STRIP)
      EXECUTE_PROCESS(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnlrbhwd/scm-1/lib/libcustnlr.so")
    ENDIF(CMAKE_INSTALL_DO_STRIP)
  ENDIF()
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(CMAKE_INSTALL_COMPONENT)
  SET(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
ELSE(CMAKE_INSTALL_COMPONENT)
  SET(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
ENDIF(CMAKE_INSTALL_COMPONENT)

FILE(WRITE "/home/jwyang/Researches/lrgan/stnlrbhwd/build/${CMAKE_INSTALL_MANIFEST}" "")
FOREACH(file ${CMAKE_INSTALL_MANIFEST_FILES})
  FILE(APPEND "/home/jwyang/Researches/lrgan/stnlrbhwd/build/${CMAKE_INSTALL_MANIFEST}" "${file}\n")
ENDFOREACH(file)
