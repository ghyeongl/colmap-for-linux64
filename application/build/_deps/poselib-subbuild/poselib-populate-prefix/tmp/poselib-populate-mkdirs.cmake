# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/_deps/poselib-src"
  "/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/_deps/poselib-build"
  "/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/_deps/poselib-subbuild/poselib-populate-prefix"
  "/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/_deps/poselib-subbuild/poselib-populate-prefix/tmp"
  "/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/_deps/poselib-subbuild/poselib-populate-prefix/src/poselib-populate-stamp"
  "/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/_deps/poselib-subbuild/poselib-populate-prefix/src"
  "/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/_deps/poselib-subbuild/poselib-populate-prefix/src/poselib-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/_deps/poselib-subbuild/poselib-populate-prefix/src/poselib-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/_deps/poselib-subbuild/poselib-populate-prefix/src/poselib-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
