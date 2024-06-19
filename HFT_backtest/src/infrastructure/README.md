# AlphaOne

# 1. C++ Style Guide
* Follow Google C++ Style Guide (https://google.github.io/styleguide/cppguide.html) mainly
* Do not use `using namesapce std;`
* Name folders as `alphaone_folder`, but name folders for applications as `AlphaOneApplication`
* Name files as `AlphaOneFile.h`
* Name classes as `AlphaOneClass`
* Name class members as `alphaone_member_`
* Name structs as `AlphaOneStruct`
* Name struct members as either `alphaone_member_` or `alphaone_member`
* Name variables as `alphaone_variable`
* Name functions as `AlphaOneFunction`
* Name enum either either as `AlphaOneEnum` or `ALPHAONE_ENUM`
* Declare enum as `enum class` but not `enum`
* Use `#ifndef _SOURCEFILENAME_H_` instead of `#pragma once`
* Use `nullptr` instead of `NULL` or `0` when initializing a pointer as a null pointer
* When handling files
    *  Name the directory that holds a certain file as `root` or `folder`, e.g. `file_root = /home/alphaone/`
    *  Name the name of a certain file as `name`, e.g. `file_name = alphaone_config.json`
    *  Name the full name of a certain file as `path`, e.g. `file_path = /home/alphaone/alphaone_config.json`
    *  `path = root + name`
    *  End `root` or `folder` with a `/`
    *  Prevent starting `name` with a `/` (since there is supposed to be a `/` at the end of `root` or `folder`)



# 2. System Architecture
<tr><td align="center"><img src="http://10.218.4.22/tzuming.kuo/alphaone/raw/master/SystemArchitecture.png" width="50%" height="50%"></tr></td>



# 3.MarketData.bin Format
See common\typedef\Typedefs.h MarketDataFileStruct