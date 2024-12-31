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
#include "atcoderlib.hpp"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace iopddl {
  // xorshift is waaaaaay fast if we do not need good-quality randomness but still want not-bad quality.
  uint64_t xor64()
  {
    static uint64_t state = time(NULL);
    uint64_t x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return state = x;
  }

  // operators for Range-Add Range-Max query segtree.
  Usage op(Usage a, Usage b) { return std::max(a, b); }
  Usage e() { return 0; }
  Usage mapping(Usage f, Usage x) { return f + x; }
  Usage composition(Usage f, Usage g) { return f + g; }
  Usage id() { return 0; }


absl::StatusOr<Solution> Solver::Solve(const Problem& _problem,
                                       absl::Duration timeout) {
  const absl::Time start_time = absl::Now();
  TotalCost best_cost;
  std::optional<Solution> best_solution;

  // We modify problem instance for simpler expression.
  ProblemInstance problem = convertToProblemInstance(_problem);

  // Initialize solution
  Solution solution;
  solution.reserve(problem.vertexs.size());
  for(int i = 0; i < problem.vertexs.size(); i++) {
    const Vertex& node = problem.vertexs[i];
    int min_itr = 0;
    for(int i = 0; i < node.selectables.size(); i++) {
      if(node.selectables[i].usage < node.selectables[min_itr].usage) {
        min_itr = i;
      }
    }
    int good_min_itr = -1;
    for(int j=0;j < node.selectables.size();j++) {
      TotalCost now_cost = 0;
      for(auto x: node.selectables[j].connection_costs) {
        if(x.first < i){
          now_cost += x.second[solution[x.first]];
        }
      }
      if(now_cost > 100032693158403LL){
        continue;
      }
      if(good_min_itr == -1 || node.selectables[j].usage < node.selectables[good_min_itr].usage){
        good_min_itr = j;
      }
    }
    if(good_min_itr != -1){
      min_itr = good_min_itr;
    }
    solution.push_back(min_itr);
  }
  best_solution = solution;
  best_cost = FastEvaluate(problem, solution).value();
  std::cout << "# Initial solution cost: " << best_cost << std::endl;

  int max_time = 0;
  for (const Vertex& node : problem.vertexs) {
    max_time = std::max(max_time, (int)node.interval.second);
  }
  atcoder::lazy_segtree<Usage, op, e, Usage, mapping, composition, id> seg(max_time);
  for (int i = 0; i < max_time; i++) {
    seg.set(i, 0);
  }
  for (int i = 0; i < problem.vertexs.size(); i++) {
    seg.apply(problem.vertexs[i].interval.first, problem.vertexs[i].interval.second, problem.vertexs[i].selectables[solution[i]].usage);
  }


  while (absl::Now() - start_time < timeout) {
    int node = xor64() % problem.vertexs.size();
    const Vertex& vertex = problem.vertexs[node];

    int selection = xor64() % vertex.selectables.size();
    const int old = solution[node];
    if(solution[node] == selection) {
      continue;
    }

    Usage usage_diff = vertex.selectables[selection].usage - vertex.selectables[old].usage;
    seg.apply(vertex.interval.first, vertex.interval.second, usage_diff);
    if(seg.all_prod() > *problem.usage_limit) {
      // revert.
      seg.apply(vertex.interval.first, vertex.interval.second, -usage_diff);
      continue;
    }

    // Recalculate cost difference with explicit old_cost and new_cost
    const Selection &oldSelection = vertex.selectables[old];
    TotalCost old_cost = oldSelection.single_cost;
    for(int i = 0; i < oldSelection.connection_costs.size(); i++) {
      int targetVertexIdx = oldSelection.connection_costs[i].first;
      old_cost += oldSelection.connection_costs[i].second[solution[targetVertexIdx]];
    }

    const Selection &nextSelection = vertex.selectables[selection];
    TotalCost new_cost = nextSelection.single_cost;
    for(int i = 0; i < nextSelection.connection_costs.size(); i++) {
      int targetVertexIdx = nextSelection.connection_costs[i].first;
      new_cost += nextSelection.connection_costs[i].second[solution[targetVertexIdx]];
    }
    TotalCost cost_diff = new_cost - old_cost;

    if(cost_diff > 0 && (cost_diff >= best_cost / 100000 || xor64() % 1000 <= 990) && (xor64() % 1000000 != 0)){
      // revert.
      seg.apply(vertex.interval.first, vertex.interval.second, -usage_diff);
      continue;
    }

    solution[node] = selection;
    best_cost = best_cost + cost_diff;
    best_solution = solution;
    std::cout << "# Found solution with cost: " << best_cost << std::endl;
  }
  if (!best_solution) {
    return absl::NotFoundError("No solution found");
  }
  return *best_solution;
}

}  // namespace iopddl
