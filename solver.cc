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

#include "solver.h"

#include <stdlib.h>

#include <iostream>
#include <optional>

#include "iopddl.h"
#include "kotamanegi_structs.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace iopddl {

absl::StatusOr<Solution> Solver::Solve(const Problem& _problem,
                                       absl::Duration timeout) {
  const absl::Time start_time = absl::Now();
  std::optional<TotalCost> best_cost;
  std::optional<Solution> best_solution;
  unsigned int seed = 2025;

  // We modify problem instance for simpler expression.
  ProblemInstance problem = convertToProblemInstance(_problem);

  while (absl::Now() - start_time < timeout) {
    Solution solution;
    solution.reserve(problem.vertexs.size());

    int counter = 0;
    for (const Vertex& node : problem.vertexs) {
      int min_itr = 0;
      counter += node.selectables.size();
      for(int i = 0; i < node.selectables.size(); i++) {
        if(node.selectables[i].usage < node.selectables[min_itr].usage) {
          min_itr = i;
        }
      }
      solution.push_back(min_itr);
    }
    auto cost = FastEvaluate(problem, solution);
    if (!cost.ok() || (best_cost && *best_cost <= *cost)) {
      continue;
    }
    std::cout << "# Found solution [" << absl::StrJoin(solution, ", ")
              << "] with cost " << *cost << std::endl;;
    
    best_cost = *cost;
    best_solution = solution;
  }
  if (!best_solution) {
    return absl::NotFoundError("No solution found");
  }
  return *best_solution;
}

}  // namespace iopddl
