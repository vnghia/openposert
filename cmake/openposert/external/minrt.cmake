if(TARGET minrt)
  return()
endif()

include(FetchContent)
FetchContent_Declare(
  minrt
  GIT_REPOSITORY https://github.com/vnghia/minrt.git
  GIT_TAG main
)
FetchContent_MakeAvailable(minrt)
