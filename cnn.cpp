// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn.hpp"

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <inference_engine.hpp>

using namespace InferenceEngine;
