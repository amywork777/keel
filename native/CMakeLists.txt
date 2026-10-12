# Native build of NVIDIA's usd-brep kernel (SMLib + SM_API) and its Python module
# (_omni_solid), without NVIDIA's Packman/premake tooling. Use native/build.sh.
#
# With -DUSD_ROOT=<OpenUSD install prefix> it also builds NVIDIA's USD layer
# (BREP_USD_DATA, BREP_SM_USD, SM_API_USD) and the real `usd` submodule; without it,
# usd_stub.cpp registers an empty `usd` submodule so everything else still works.
cmake_minimum_required(VERSION 3.24)
project(keel_native CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(USD_BREP_SRC "" CACHE PATH "Checkout of NVIDIA-Omniverse/usd-brep")
if(NOT EXISTS "${USD_BREP_SRC}/source/SMLib")
  message(FATAL_ERROR "Set -DUSD_BREP_SRC to a usd-brep checkout")
endif()
set(SRC "${USD_BREP_SRC}/source")

# NVIDIA's macOS warning set (premake5-utils.lua) plus their non-Windows defines.
set(KERNEL_WARNINGS -Wno-missing-braces -Wno-write-strings -Wno-deprecated -w)

file(GLOB SMLIB_SOURCES "${SRC}/SMLib/src/*.cpp")
list(FILTER SMLIB_SOURCES EXCLUDE REGEX "StdAfx\\.cpp$")
add_library(SMLib STATIC ${SMLIB_SOURCES})
target_include_directories(SMLib PUBLIC "${SRC}/SMLib/inc")
target_compile_options(SMLib PRIVATE ${KERNEL_WARNINGS})

file(GLOB SMAPI_SOURCES "${SRC}/SM_API/src/*.cpp")
list(FILTER SMAPI_SOURCES EXCLUDE REGEX "StdAfx\\.cpp$")
add_library(SM_API STATIC ${SMAPI_SOURCES})
target_include_directories(SM_API PUBLIC "${SRC}/SM_API/inc")
target_compile_definitions(SM_API PRIVATE SM_API_EXPORTS)
target_compile_options(SM_API PRIVATE ${KERNEL_WARNINGS})
target_link_libraries(SM_API PUBLIC SMLib)

find_package(Python 3.12 EXACT REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
                OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)

set(USD_ROOT "" CACHE PATH "OpenUSD install prefix (optional)")
file(GLOB PY_SOURCES "${SRC}/SmPyLib/src/*.cpp")

if(USD_ROOT)
  find_package(pxr CONFIG REQUIRED PATHS "${USD_ROOT}" NO_DEFAULT_PATH)
  set(USD_LIBS ar arch gf js kind pcp plug sdf sdr tf trace usd usdGeom usdShade usdUtils vt work)

  function(usd_layer name)
    file(GLOB layer_sources "${SRC}/${name}/src/*.cpp")
    list(FILTER layer_sources EXCLUDE REGEX "StdAfx\\.cpp$")
    add_library(${name} STATIC ${layer_sources})
    target_include_directories(${name} PUBLIC "${SRC}/${name}/inc")
    target_compile_definitions(${name} PRIVATE ${ARGN} TBB_SUPPRESS_DEPRECATED_MESSAGES)
    target_compile_options(${name} PRIVATE ${KERNEL_WARNINGS})
    target_link_libraries(${name} PUBLIC ${USD_LIBS})
  endfunction()

  usd_layer(BREP_USD_DATA USDBREP_ENABLE_EXPORTS)
  usd_layer(BREP_SM_USD SMU_ENABLE_EXPORTS)
  target_link_libraries(BREP_SM_USD PUBLIC SMLib BREP_USD_DATA)
  usd_layer(SM_API_USD SM_API_USD_EXPORTS)
  target_link_libraries(SM_API_USD PUBLIC SM_API BREP_SM_USD BREP_USD_DATA)

  pybind11_add_module(_omni_solid ${PY_SOURCES})
  target_link_libraries(_omni_solid PRIVATE SM_API_USD)
  set_target_properties(_omni_solid PROPERTIES BUILD_RPATH "${USD_ROOT}/lib")
  # The omniSolid schema plugin the USD layer registers at runtime (OMNISOLID_PLUGIN_PATH).
  add_custom_command(TARGET _omni_solid POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${SRC}/schema/omniSolid/resources"
            "$<TARGET_FILE_DIR:_omni_solid>/omniSolid/resources")
else()
  list(FILTER PY_SOURCES EXCLUDE REGEX "SmPyUsd\\.cpp$")
  pybind11_add_module(_omni_solid ${PY_SOURCES} usd_stub.cpp)
  target_link_libraries(_omni_solid PRIVATE SM_API)
endif()
target_include_directories(_omni_solid PRIVATE "${SRC}/SmPyLib/src")
target_compile_options(_omni_solid PRIVATE ${KERNEL_WARNINGS})
