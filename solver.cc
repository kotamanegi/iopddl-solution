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
#include <map>
#include <optional>
#include <fstream>
#include <set>

#include "iopddl.h"
#include "kotamanegi_structs.h"
#include "atcoderlib.hpp"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace iopddl
{
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
  TotalUsage op(TotalUsage a, TotalUsage b) { return std::max(a, b); }
  TotalUsage e() { return 0; }
  TotalUsage mapping(TotalUsage f, TotalUsage x) { return f + x; }
  TotalUsage composition(TotalUsage f, TotalUsage g) { return f + g; }
  TotalUsage id() { return 0; }

  struct Operation
  {
    signed long node;
    signed long new_selection;
  };

  class State
  {
  public:
    ProblemInstance problem;
    bool isValid;
    Solution solution;
    atcoder::lazy_segtree<TotalUsage, op, e, TotalUsage, mapping, composition, id> seg;
    TotalCost current_cost;
    TotalUsage total_memory_usage;
    long long seg_size;

    State(ProblemInstance problem, bool isValid, Solution solution, atcoder::lazy_segtree<TotalUsage, op, e, TotalUsage, mapping, composition, id> seg, TotalCost current_cost, TotalUsage total_memory_usage, long long seg_size)
    {
      this->problem = problem;
      this->isValid = isValid;
      this->solution = solution;
      this->seg = seg;
      this->current_cost = current_cost;
      this->total_memory_usage = total_memory_usage;
      this->seg_size = seg_size;
    }
    void update(std::vector<Operation> operations)
    {
      for (Operation operation : operations)
      {
        const Vertex &vertex = problem.vertexs[operation.node];
        const int old = solution[operation.node];
        if (old == operation.new_selection)
        {
          continue;
        }

        // memory
        Usage usage_diff = vertex.selectables[operation.new_selection].usage - vertex.selectables[old].usage;
        seg.apply(vertex.interval.first, vertex.interval.second, usage_diff);
        total_memory_usage += usage_diff * (vertex.interval.second - vertex.interval.first);
        // calculate the difference of score.

        // Recalculate cost difference with explicit old_cost and new_cost
        const Selection &oldSelection = vertex.selectables[old];
        TotalCost old_cost = oldSelection.single_cost;
        for (int i = 0; i < oldSelection.connection_costs.size(); i++)
        {
          int targetVertexIdx = oldSelection.connection_costs[i].first;
          old_cost += oldSelection.connection_costs[i].second[solution[targetVertexIdx]];
        }

        const Selection &nextSelection = vertex.selectables[operation.new_selection];
        TotalCost new_cost = nextSelection.single_cost;
        for (int i = 0; i < nextSelection.connection_costs.size(); i++)
        {
          int targetVertexIdx = nextSelection.connection_costs[i].first;
          new_cost += nextSelection.connection_costs[i].second[solution[targetVertexIdx]];
        }
        TotalCost cost_diff = new_cost - old_cost;
        current_cost = current_cost + cost_diff;
        solution[operation.node] = operation.new_selection;
      }
      check_validaty();
    }
    TotalCost get_cost()
    {
      return current_cost;
    }
    void check_validaty()
    {
      isValid = true;
      if (seg.all_prod() > *problem.usage_limit)
      {
        isValid = false;
      }
    }
  };

  State create_initial_state(
      const ProblemInstance &problem)
  {
    Solution solution;
    solution.reserve(problem.vertexs.size());
    for (int i = 0; i < problem.vertexs.size(); i++)
    {
      const Vertex &node = problem.vertexs[i];
      int min_itr = 0;
      for (int i = 0; i < node.selectables.size(); i++)
      {
        if (node.selectables[i].usage < node.selectables[min_itr].usage)
        {
          min_itr = i;
        }
      }
      solution.push_back(min_itr);
    }
    TotalCost cost = FastEvaluate(problem, solution).value();

    int max_time = 0;
    for (const Vertex &node : problem.vertexs)
    {
      max_time = std::max(max_time, (int)node.interval.second);
    }
    atcoder::lazy_segtree<TotalUsage, op, e, TotalUsage, mapping, composition, id> seg(max_time);
    for (int i = 0; i < max_time; i++)
    {
      seg.set(i, 0);
    }
    TotalUsage memory_cost = 0;
    for (int i = 0; i < problem.vertexs.size(); i++)
    {
      seg.apply(problem.vertexs[i].interval.first, problem.vertexs[i].interval.second, problem.vertexs[i].selectables[solution[i]].usage);
      memory_cost += problem.vertexs[i].selectables[solution[i]].usage * (problem.vertexs[i].interval.second - problem.vertexs[i].interval.first);
    }

    return State{problem, true, solution, seg, cost, memory_cost, max_time};
  }

  absl::StatusOr<Solution> Solver::Solve(const Problem &_problem,
                                         absl::Duration timeout)
  {
    const absl::Time start_time = absl::Now();
    TotalCost global_best_cost = std::numeric_limits<TotalCost>::max();
    std::optional<Solution> global_best_solution;

    // We modify problem instance for simpler expression.
    ProblemInstance problem = convertToProblemInstance(_problem);

    const long long TICK = 1000000000000000000;
    std::map<int, int> update_counter;

    while (absl::Now() - start_time < timeout)
    {
      TotalCost cur_best_cost = std::numeric_limits<TotalCost>::max();
      std::optional<Solution> cur_best_solution;

      // Initialize solution
      State state = create_initial_state(problem);

      absl::Time last_updated = absl::Now();

      absl::Time last_output = absl::Now();
      while (absl::Now() - start_time < timeout && absl::Now() - last_updated < absl::Seconds(5))
      {
        const int current_prob = xor64() % 1000;
        if (current_prob > 0.8 * 1000)
        {
          int node = xor64() % problem.vertexs.size();
          const Vertex &vertex = problem.vertexs[node];

          int selection = xor64() % vertex.selectables.size();
          const int old = state.solution[node];
          if (state.solution[node] == selection)
          {
            continue;
          }
          TotalCost old_cost = state.get_cost();
          state.update({Operation{node, selection}});
          if (!state.isValid)
          {
            // memory fail.
            state.update({Operation{node, old}});
            continue;
          }

          TotalCost new_cost = state.get_cost();
          TotalCost cost_diff = new_cost - old_cost;

          if (cost_diff > 0 && (cost_diff >= state.current_cost / 100000 || xor64() % 1000 <= 990) && (xor64() % 1000000 != 0))
          {
            // revert.
            state.update({Operation{node, old}});
            continue;
          }
        }
        else
        {
          int node = xor64() % problem.vertexs.size();
          const Vertex &vertex = problem.vertexs[node];

          int selection = xor64() % vertex.selectables.size();
          const int old = state.solution[node];
          if (state.solution[node] == selection)
          {
            continue;
          }

          std::vector<Operation> targets;
          std::vector<Operation> prevs;
          targets.push_back(Operation{node, selection});
          prevs.push_back(Operation{node, old});

          for (const auto &x : vertex.selectables[selection].connection_costs)
          {
            int target_node = x.first;
            std::vector<int> cost_oks;
            bool no_fix = false;
            int priority = 100;
            for (int j = 0; j < x.second.size(); ++j)
            {
              int go = x.second[j] / (TICK / 10);
              if (go < priority)
              {
                priority = go;
                no_fix = false;
                cost_oks.clear();
              }
              if (go == priority)
              {
                if (state.solution[target_node] == j)
                {
                  no_fix = true;
                }
                cost_oks.push_back(j);
              }
            }
            if (no_fix)
            {
              continue;
            }
            if (priority != 0)
            {
              continue;
            }
            int target_selection = cost_oks[0];
            for (int j = 1; j < cost_oks.size(); ++j)
            {
              if (problem.vertexs[target_node].selectables[cost_oks[j]].usage < problem.vertexs[target_node].selectables[target_selection].usage)
              {
                target_selection = cost_oks[j];
              }
            }
            targets.push_back(Operation{target_node, target_selection});
            prevs.push_back(Operation{target_node, state.solution[target_node]});
          }

          TotalCost old_cost = state.get_cost();
          state.update(targets);
          if (!state.isValid)
          {
            state.update(prevs);
            continue;
          }
          TotalCost new_cost = state.get_cost();
          TotalCost cost_diff = new_cost - old_cost;

          if (cost_diff > 0 && (cost_diff >= state.current_cost / 100000 || xor64() % 1000 <= 995) && (xor64() % 1000000 != 0))
          {
            // revert.
            state.update(prevs);
            continue;
          }
          update_counter[targets.size()]++;
        }
        if (state.current_cost < cur_best_cost)
        {
          cur_best_cost = state.current_cost;
          cur_best_solution = state.solution;
          auto current = absl::Now();
          last_updated = absl::Now();
          if (current - last_output > absl::Seconds(1))
          {
            last_output = current;
            std::cout << "# Found solution with cost: " << cur_best_cost << std::endl;
          }
        }
      }
      TotalCost single_cost = 0;
      for (int i = 0; i < cur_best_solution->size(); i++)
      {
        single_cost += problem.vertexs[i].selectables[(*cur_best_solution)[i]].single_cost;
      }
      std::cout << "# Single cost: " << single_cost << std::endl;
      std::cout << "# connection_cost: " << cur_best_cost - single_cost << std::endl;
      std::cout << "# Final cost: " << cur_best_cost << std::endl;
      auto original_evaluate = Evaluate(_problem, *cur_best_solution);
      if (original_evaluate.ok())
      {
        if (global_best_cost > original_evaluate.value())
        {
          std::cout << "# Update: " << original_evaluate.value() << std::endl;
          global_best_cost = original_evaluate.value();
          global_best_solution = cur_best_solution;
        }
      }
    }

    std::cout << "# Final cost: " << global_best_cost << std::endl;
    for (auto x : update_counter)
    {
      std::cout << "# counters: " << x.first << " " << x.second << std::endl;
    }
    if (!global_best_solution)
    {
      return absl::NotFoundError("No solution found");
    }
    return *global_best_solution;
  }

} // namespace iopddl
