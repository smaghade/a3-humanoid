# C++ port of the A3 humanoid simulation (same behaviour as the Python version).
#   mkdir build && cd build
#   cmake .. -DMUJOCO_DIR=$HOME/mujoco-3.15.0
#   make -j
cmake_minimum_required(VERSION 3.16)
project(a3_cpp LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(MUJOCO_DIR "$ENV{HOME}/mujoco-3.15.0" CACHE PATH "Folder of the unpacked MuJoCo release")
find_path(MUJOCO_INCLUDE mujoco/mujoco.h HINTS ${MUJOCO_DIR}/include REQUIRED)
find_library(MUJOCO_LIB mujoco HINTS ${MUJOCO_DIR}/lib REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs dnn)
find_package(yaml-cpp REQUIRED)

add_library(a3core STATIC
  src/config.cpp src/robot.cpp src/safety.cpp src/scene.cpp src/course.cpp
  src/render.cpp src/viewer.cpp src/detector.cpp src/spatial.cpp
  src/obstacles.cpp src/tracker.cpp src/grid_planner.cpp src/arm.cpp src/util.cpp)
target_include_directories(a3core PUBLIC include ${MUJOCO_INCLUDE} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(a3core PUBLIC ${MUJOCO_LIB} glfw ${OpenCV_LIBS} yaml-cpp)
get_filename_component(MUJOCO_LIBDIR ${MUJOCO_LIB} DIRECTORY)
set(CMAKE_BUILD_RPATH ${MUJOCO_LIBDIR})
set(CMAKE_INSTALL_RPATH ${MUJOCO_LIBDIR})

foreach(app task_demo pick_demo play_sim mission_sim plot_log demo_mock_walk
            detection_test finder_test export_joint_limits camera_view)
  add_executable(${app} apps/${app}.cpp)
  target_link_libraries(${app} PRIVATE a3core)
endforeach()
add_executable(test_safety tests/test_safety.cpp)
target_link_libraries(test_safety PRIVATE a3core)
enable_testing()
add_test(NAME test_safety COMMAND test_safety WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..)
