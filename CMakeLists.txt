cmake_minimum_required(VERSION 3.17)

project("DialogVideoGenerator"
        VERSION 1.1.0
        DESCRIPTION "Dialog video/picture generator"
        LANGUAGES CXX CUDA
        )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
if (DEFINED Common613_ROOT_DIR)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${Common613_ROOT_DIR}/cmake")
endif()
if (COMMON613_STACKTRACE_DEBUG)
    add_definitions(-DCOMMON613_STACKTRACE_DEBUG=1)
endif()

cmake_policy(SET CMP0104 OLD)
find_package(CUDAToolkit 10 REQUIRED)

find_package(Common613 1.1.0 REQUIRED)

set(png++_ROOT_DIR "thirdparty/png++-0.2.9")
find_package(png++ REQUIRED)
if (WIN32)
    add_definitions(-D__STDC_LIB_EXT1__)
endif ()

set(TinyUTF8_ROOT_DIR "thirdparty/tiny-utf8")
find_package(TinyUTF8 REQUIRED)

find_package(fmt REQUIRED)

find_package(Freetype REQUIRED)

if (NOT DIALOG_VIDEO_GENERATOR_DISABLE_SAVE_VIDEO_IPC_STRATEGY)
#    find_package(Boost REQUIRED COMPONENTS headers) # process
    add_definitions(-DDIALOGVIDEOGENERATOR_ENABLE_SAVE_VIDEO_IPC_STRATEGY)
    link_libraries(Boost::headers)
endif ()
if (MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /utf-8")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"/utf-8 /std:c++14\"")
endif (MSVC)

set(${PROJECT_NAME}_HEADERS
        common.h # common
        math/pos_utils.h math/pos_arith.h # pos utils
        math/random_utils.h # random utils
        cuda/cuda_utils.h # cuda support
        math/time_utils.h # other utils
        text/text_render_utils.h text/text_render_details.h # text rendering support
        image.h # basic images and sprites
        drawable.h drawables/movements.h drawables/alpha_changes.h # drawables
        drawables/filters.h
        engine.h # engine facade
        )

set(${PROJECT_NAME}_SOURCES
        common.cpp
        utils/random_utils.cpp
        cuda/cuda_utils.cpp cuda/cuda_images.cu
        utils/text_render_utils.cpp
        image.cpp
        drawables/drawable.cpp drawables/stand.cpp drawables/text_like.cpp drawables/animated.cpp
        drawables/movements.cpp drawables/alpha_changes.cpp
        drawables/filters.cpp
        engine.cpp
        engine_strategies/frame_strategies.cpp engine_strategies/streaming_strategy.cpp engine_strategies/child_proc_strategy.cpp
        )

set(${PROJECT_NAME}_FACADE_HEADERS
        abstraction/character.h abstraction/character_to_draw.h # multiscene
        abstraction/position.h # abstract position
        abstraction/action_animations.h abstraction/dialog.h
        abstraction/common_filters.h
#        abstraction/director.h # director facade
        )

set(${PROJECT_NAME}_FACADE_SOURCES
        abstraction/character.cpp abstraction/character_to_draw.cpp
        abstraction/action_animations.cpp abstraction/dialog.cpp
        abstraction/common_filters.cpp
#        abstraction/director.cpp
        )

list(TRANSFORM ${PROJECT_NAME}_HEADERS PREPEND "include/dialog_video_generator/" OUTPUT_VARIABLE ${PROJECT_NAME}_HEADERS_RELATIVE)
list(TRANSFORM ${PROJECT_NAME}_SOURCES PREPEND "src/" OUTPUT_VARIABLE ${PROJECT_NAME}_SOURCES_RELATIVE)
list(TRANSFORM ${PROJECT_NAME}_FACADE_HEADERS PREPEND "include/dialog_video_generator/" OUTPUT_VARIABLE ${PROJECT_NAME}_FACADE_HEADERS_RELATIVE)
list(TRANSFORM ${PROJECT_NAME}_FACADE_SOURCES PREPEND "src/" OUTPUT_VARIABLE ${PROJECT_NAME}_FACADE_SOURCES_RELATIVE)

add_library(${PROJECT_NAME} STATIC
        ${${PROJECT_NAME}_HEADERS_RELATIVE} ${${PROJECT_NAME}_SOURCES_RELATIVE})
target_include_directories(
        ${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(${PROJECT_NAME} PUBLIC
        ${png++_INCLUDE_DIRS}
        ${TinyUTF8_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PUBLIC
        CUDA::cudart
        CUDA::cudart_static
        CUDA::cuda_driver
        ${png++_LIBRARIES}
        Freetype::Freetype
        ${TinyUTF8_LIBRARIES}
        Common613::Common613
)

add_library(${PROJECT_NAME}_FACADE STATIC
        ${${PROJECT_NAME}_FACADE_HEADERS_RELATIVE} ${${PROJECT_NAME}_FACADE_SOURCES_RELATIVE})
target_include_directories(
        ${PROJECT_NAME}_FACADE PUBLIC
        $<BUILD_INTERFACE:${DialogVideoGenerator_SOURCE_DIR};${DialogVideoGenerator_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(${PROJECT_NAME}_FACADE PUBLIC
        ${Common613_INCLUDE_DIRS}
        ${png++_INCLUDE_DIRS}
        ${TinyUTF8_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_FACADE PUBLIC ${PROJECT_NAME})

add_subdirectory("test")
add_subdirectory("doc")
