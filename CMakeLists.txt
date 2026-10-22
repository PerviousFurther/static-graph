cmake_minimum_required(VERSION 3.20)
project(static_graph LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Header-only: core and exec need nothing else; gfx/rdg is API independent.
add_library(sg INTERFACE)
target_include_directories(sg INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/inc)

enable_testing()

function(sg_test name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE sg)
  if(MSVC)
    target_compile_options(${name} PRIVATE /W4 /permissive-)
  else()
    target_compile_options(${name} PRIVATE -Wall -Wextra)
  endif()
  add_test(NAME ${name} COMMAND ${name})
endfunction()

sg_test(quick_start examples/quick_start.cpp)
sg_test(connection_activation examples/connection_activation.cpp)
sg_test(connection_constructor examples/connection_constructor.cpp)
sg_test(connection_function examples/connection_function.cpp)
sg_test(core_mixin tests/core/mixin.cpp)
sg_test(exec tests/exec/exec.cpp)
find_package(Threads REQUIRED)
target_link_libraries(exec PRIVATE Threads::Threads)
# Each component header must compile alone.
file(GLOB_RECURSE sg_headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/inc
  CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/inc/sg/*.hpp)
foreach(h IN LISTS sg_headers)
  string(MAKE_C_IDENTIFIER ${h} id)
  set(src ${CMAKE_CURRENT_BINARY_DIR}/header_check/${id}.cpp)
  file(CONFIGURE OUTPUT ${src} CONTENT "#include <${h}>\n")
  list(APPEND sg_header_sources ${src})
endforeach()
add_library(sg_header_check OBJECT ${sg_header_sources})
target_link_libraries(sg_header_check PRIVATE sg)

sg_test(exec_adaptors tests/exec/adaptors.cpp)
sg_test(exec_bulk tests/exec/bulk.cpp)
sg_test(exec_scope tests/exec/scope.cpp)
sg_test(exec_task tests/exec/task.cpp)
# GCC flags task's allocator-aware promise new/delete pair, which is the
# pattern C++26 task specifies, as mismatched.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(exec_task PRIVATE -Wno-mismatched-new-delete)
endif()
sg_test(rdg_graph tests/gfx/rdg_graph.cpp)
sg_test(rdg_hazards tests/gfx/rdg_hazards.cpp)
