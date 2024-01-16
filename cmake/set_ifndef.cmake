# /**************************************************************
#  * @Author: lijinwen 
#  * @Date: 2021-08-29 10:14:11  
#  * @Last Modified by: lijinwen 
#  * @Last Modified time: 2021-09-09 13:50:15 
#  **************************************************************/

function (set_ifndef variable value)
  if(NOT DEFINED ${variable})
    set(${variable} ${value} PARENT_SCOPE)
  endif()
  
endfunction()
