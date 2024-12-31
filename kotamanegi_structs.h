/*
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef IOPDDL_KOTAMANEGI_STRUCTS_H_
#define IOPDDL_KOTAMANEGI_STRUCTS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "iopddl.h"
#include "absl/numeric/int128.h"
#include "absl/status/statusor.h"

// This is a simple selection/projection problem, right?
namespace iopddl {

using Cost = int64_t;
using Usage = int64_t;
using TimeIdx = int64_t;
using VertexIdx = int64_t;
using SelectionIdx = int64_t;
using Interval = std::pair<TimeIdx, TimeIdx>;
using Solution = std::vector<SelectionIdx>;
using TotalUsage = absl::int128;
using TotalCost = absl::int128;

struct Selection {
  Cost single_cost;
  std::vector<std::pair<VertexIdx, std::vector<Cost>>> connection_costs; // cost = connection_costs[x][solution[connection_costs[x].first]];
  Usage usage;
  bool operator==(const Selection& other) const = default;
};

struct Vertex {
  Interval interval;  // Interpreted as half-open with an exclusive upper bound
  std::vector<Selection> selectables;
  bool operator==(const Vertex& other) const = default;
};

struct ProblemInstance {
  std::string name;
  std::vector<Vertex> vertexs;
  std::optional<Usage> usage_limit;
  bool operator==(const ProblemInstance& other) const = default;
};

ProblemInstance convertToProblemInstance(const Problem& problem);
ProblemInstance zipTimeInterval(const ProblemInstance &problem);
absl::StatusOr<TotalCost> FastEvaluate(const ProblemInstance& problem,
                                   const Solution& solution);
}  // namespace iopddl

#endif  // IOPDDL_IOPDDL_KOTAMANEGI_STRUCTS_H_
