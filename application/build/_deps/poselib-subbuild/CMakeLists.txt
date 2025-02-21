# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.25.2)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(poselib-populate NONE)



include(ExternalProject)
ExternalProject_Add(poselib-populate
                     "UPDATE_DISCONNECTED" "False" "DOWNLOAD_EXTRACT_TIMESTAMP" "FALSE" "URL" "https://github.com/PoseLib/PoseLib/archive/8028473d92c9347794a0e3d3541863b5cbb15743.zip" "URL_HASH" "SHA256=20ff75d14c839dcab3d38fb51503c47088898f394b812472a758ec17720a54c2"
                    SOURCE_DIR          "/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/_deps/poselib-src"
                    BINARY_DIR          "/home/ghryu/Drives/Env-AI/gaussian-splatting/colmap/build/_deps/poselib-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)


