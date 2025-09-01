# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_servo_keyboard_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED servo_keyboard_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(servo_keyboard_FOUND FALSE)
  elseif(NOT servo_keyboard_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(servo_keyboard_FOUND FALSE)
  endif()
  return()
endif()
set(_servo_keyboard_CONFIG_INCLUDED TRUE)

# output package information
if(NOT servo_keyboard_FIND_QUIETLY)
  message(STATUS "Found servo_keyboard: 0.0.0 (${servo_keyboard_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'servo_keyboard' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${servo_keyboard_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(servo_keyboard_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${servo_keyboard_DIR}/${_extra}")
endforeach()
