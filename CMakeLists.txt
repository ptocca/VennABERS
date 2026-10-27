cmake_minimum_required(VERSION 3.18...3.31)
project(fast_ivap LANGUAGES CXX)

# Header-only C++ library: fast_ivap::fast_ivap
add_library(fast_ivap INTERFACE)
add_library(fast_ivap::fast_ivap ALIAS fast_ivap)
target_include_directories(
  fast_ivap INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                      $<INSTALL_INTERFACE:include>)
target_compile_features(fast_ivap INTERFACE cxx_std_20)

if(SKBUILD)
  # Python extension module fast_ivap._core, built by scikit-build-core.
  find_package(Python 3.11 REQUIRED
               COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_core STABLE_ABI NB_STATIC src/bindings/core.cpp)
  target_link_libraries(_core PRIVATE fast_ivap)
  install(TARGETS _core LIBRARY DESTINATION fast_ivap)
  return()
endif()

# Standalone C++ build: tests, example and install rules for find_package().
option(FAST_IVAP_BUILD_TESTS "Build the C++ tests" ${PROJECT_IS_TOP_LEVEL})
option(FAST_IVAP_BUILD_EXAMPLES "Build the C++ example" ${PROJECT_IS_TOP_LEVEL})

if(FAST_IVAP_BUILD_TESTS)
  enable_testing()
  add_executable(test_ivap tests/cpp/test_ivap.cpp)
  target_link_libraries(test_ivap PRIVATE fast_ivap)
  add_test(NAME test_ivap COMMAND test_ivap)
endif()

if(FAST_IVAP_BUILD_EXAMPLES)
  add_executable(ivap_example examples/cpp/example.cpp)
  target_link_libraries(ivap_example PRIVATE fast_ivap)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS fast_ivap EXPORT fast_ivapTargets)
install(EXPORT fast_ivapTargets NAMESPACE fast_ivap::
        DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/fast_ivap)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/fast_ivapConfig.cmake
     "include(\"\${CMAKE_CURRENT_LIST_DIR}/fast_ivapTargets.cmake\")\n")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fast_ivapConfig.cmake
        DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/fast_ivap)
