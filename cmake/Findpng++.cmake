# - Try to find png++
#
# The following variables are optionally searched for defaults
#  png++_ROOT_DIR:            Base directory where all GLOG components are found
#
# The following are set after configuration is done: 
#  png++_FOUND
#  png++_INCLUDE_DIRS
#  png++_LIBRARIES

find_package(PNG REQUIRED)

include(FindPackageHandleStandardArgs)

find_path(png++_INCLUDE_DIR
    NAMES
        color.hpp
        config.hpp
        consumer.hpp
        convert_color_space.hpp
        end_info.hpp
        error.hpp
        ga_pixel.hpp
        generator.hpp
        gray_pixel.hpp
        image.hpp
        image_info.hpp
        index_pixel.hpp
        info.hpp
        info_base.hpp
        io_base.hpp
        packed_pixel.hpp
        palette.hpp
        pixel_buffer.hpp
        pixel_traits.hpp
        png.hpp
        reader.hpp
        require_color_space.hpp
        rgb_pixel.hpp
        rgba_pixel.hpp
        solid_pixel_buffer.hpp
        streaming_base.hpp
        tRNS.hpp
        types.hpp
        writer.hpp
    PATHS
        ${png++_ROOT_DIR}
    )

set(png++_INCLUDE_DIRS ${png++_INCLUDE_DIR} ${PNG_INCLUDE_DIRS})
set(png++_LIBRARIES ${PNG_LIBRARIES})

find_package_handle_standard_args(png++ DEFAULT_MSG
    png++_INCLUDE_DIR)
