set (TARGET_NAME "application")

set (INCLUSION
  "RunStrategy"
)

macro(SUBDIRLIST RESULT CURDIR)
  set(DIRLIST "")

  file(GLOB CHILDREN RELATIVE ${CURDIR} ${CURDIR}/*)
  foreach(CHILD ${CHILDREN})
    if(IS_DIRECTORY ${CURDIR}/${CHILD})
      set(DIRLIST ${DIRLIST} ${CHILD})
    endif()
  endforeach()

  set(${RESULT} ${DIRLIST})
endmacro()

if(FULL)
  SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR})
  foreach(SUBDIR ${SUBDIRS})
    add_subdirectory(${SUBDIR})
  endforeach()
else()
  SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR})
  foreach(SUBDIR ${SUBDIRS})
    list(FIND INCLUSION ${SUBDIR} POSITION)
    if(${POSITION} GREATER -1)
      add_subdirectory(${SUBDIR})
    else()
      file(REMOVE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${SUBDIR})
    endif()
  endforeach()
endif()