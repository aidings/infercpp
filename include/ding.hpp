#pragma once

#define OFF 0
#define ON 1

#define USE_ONNX OFF
#define USE_TORCH OFF
#define USE_JSON ON
#define USE_NPY ON


#if USE_ONNX
#include "onnx.hpp"
#endif

#if USE_JSON
#include "json.hpp"
using Json = nlohmann::json;
#endif

#if USE_NPY
#include "npy.hpp"
#endif