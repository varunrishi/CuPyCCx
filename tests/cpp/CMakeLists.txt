include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(test_cupyccx
    test_ccd.cpp
    test_integrals.cpp
)

target_link_libraries(test_cupyccx PRIVATE
    cupyccx_core
    GTest::gtest_main
)

target_include_directories(test_cupyccx PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

include(GoogleTest)
gtest_discover_tests(test_cupyccx)
