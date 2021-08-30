# - Try to find TinyUTF8
#
# The following variables are optionally searched for defaults
#  TinyUTF8_ROOT_DIR
#
# The following are set after configuration is done: 
#  TinyUTF8_FOUND
#  TinyUTF8_INCLUDE_DIRS
# # TinyUTF8_LIBRARIES # header-only

include(FindPackageHandleStandardArgs)

find_path(TinyUTF8_INCLUDE_DIR
    NAMES
        tinyutf8/tinyutf8.h
    PATHS
        ${TinyUTF8_ROOT_DIR}/include
    )

find_package_handle_standard_args(TinyUTF8 DEFAULT_MSG
    TinyUTF8_INCLUDE_DIR)

if(TinyUTF8_FOUND)
    set(TinyUTF8_INCLUDE_DIRS ${TinyUTF8_INCLUDE_DIR})
endif()
