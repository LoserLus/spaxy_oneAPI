cmake_minimum_required (VERSION 3.4)

project(SPAXY CXX)

set(CMAKE_CXX_COMPILER icpx)

set(SOURCE_FILE spaxy.cpp)
set(TARGET_NAME spaxy)
set(COMPILE_FLAGS "-fsycl -Wall ${WIN_FLAG}")
set(LINK_FLAGS "-fsycl")
add_executable(${TARGET_NAME} ${SOURCE_FILE})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")