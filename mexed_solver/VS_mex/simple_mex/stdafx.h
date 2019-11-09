// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//
#ifndef MEX_COMPILE_FLAG //to distungish between mex-matlab and VS compilations 
#pragma once

#include "targetver.h"

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files:
//#include <windows.h>  //When compiling from matlab 'windows.h' is not included, 
					    //including it can cause various problem with macros, like syntax error with "std::min" 



// TODO: reference additional headers your program requires here
#endif