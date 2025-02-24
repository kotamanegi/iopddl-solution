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
#include <omp.h>

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
    static uint64_t state = 7236178123;
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
    ProblemInstance &problem;
    bool isValid;
    Solution solution;
    atcoder::lazy_segtree<TotalUsage, op, e, TotalUsage, mapping, composition, id> seg;
    TotalCost current_cost;
    TotalUsage total_memory_usage;
    long long seg_size;

    State(ProblemInstance &problem, bool isValid, Solution solution)
        : problem(problem),
          isValid(isValid),
          solution(solution)
    {
    }
    void dfs(int node, std::vector<int> &visited)
    {
      visited[node] = 1;
      // determine the best solution in the world.
      int target_itr = 0;
      Usage target_memory = std::numeric_limits<Usage>::max();
      const Cost TICK = 100000000000000000;
      for (int i = 0; i < problem.vertexs[node].selectables.size(); i++)
      {
        // check if valid.
        bool is_valid = 1;
        for (auto &x : problem.vertexs[node].selectables)
        {
          for (auto y : x.connection_costs)
          {
            if (visited[y.first] == 1)
            {
              if (y.second[solution[y.first]] >= TICK)
              {
                is_valid = 0;
              }
            }
          }
        }
        if (!is_valid)
        {
          continue;
        }
        Usage memory = problem.vertexs[node].selectables[i].usage;
        if (memory < target_memory)
        {
          target_memory = memory;
          target_itr = i;
        }
      }
      if (target_memory == std::numeric_limits<Usage>::max())
      {
        for (int i = 0; i < problem.vertexs[node].selectables.size(); i++)
        {
          Usage memory = problem.vertexs[node].selectables[i].usage;
          if (memory < target_memory)
          {
            target_memory = memory;
            target_itr = i;
          }
        }
      }
      update({Operation{node, target_itr}});
      for (const auto &x : problem.vertexs[node].selectables[solution[node]].connection_costs)
      {
        if (visited[x.first] == 0)
        {
          dfs(x.first, visited);
        }
      }
    }
    void initialize()
    {
      // not used anymore.
      return;
      std::vector<int> visited(problem.vertexs.size(), 0);
      int counter = 0;
      for (int i = 0; i < problem.vertexs.size(); i++)
      {
        if (visited[i] == 0)
        {
          counter++;
          dfs(i, visited);
        }
      }

      check_validaty();
      std::cout << "# counter: " << counter << std::endl;
      std::cout << "# isitOK? :" << isValid << std::endl;
      std::cout << "# initial cost: " << current_cost << std::endl;
    }
    void update(std::vector<Operation> operations)
    {
      if (operations.size() >= 300 && false)
      {
        update_recalc(operations);
      }
      else
      {
        update_diff(operations);
      }
    }
    void update_diff(std::vector<Operation> operations)
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
    void update_recalc(std::vector<Operation> operations)
    {
      std::cerr << "# update_recalc" << std::endl;
      for (Operation operation : operations)
      {
        solution[operation.node] = operation.new_selection;
      }
      construct();
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
    void construct()
    {
      current_cost = FastEvaluate(problem, solution, false).value();
      // memory
      int max_time = 0;
      for (const Vertex &node : problem.vertexs)
      {
        max_time = std::max(max_time, (int)node.interval.second + 1);
      }
      std::vector<TotalUsage> seg_data(max_time);
      for (int i = 0; i < problem.vertexs.size(); i++)
      {
        seg_data[problem.vertexs[i].interval.first] += problem.vertexs[i].selectables[solution[i]].usage;
        seg_data[problem.vertexs[i].interval.second] -= problem.vertexs[i].selectables[solution[i]].usage;
      }
      for (int i = 1; i < max_time; i++)
      {
        seg_data[i] += seg_data[i - 1];
      }
      seg = atcoder::lazy_segtree<TotalUsage, op, e, TotalUsage, mapping, composition, id>(seg_data);
      check_validaty();
    }
  };

  State create_initial_state(ProblemInstance &problem)
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
    State state = State{problem, true, solution};
    state.construct();
    return state;
  }

  absl::StatusOr<Solution> Solver::Solve(const Problem &_problem,
                                         absl::Duration _timeout)
  {
#define output(x) // std::cerr << x << std::endl
    const absl::Time start_time = absl::Now();
    std::cout << "# Entered Solver::Solve function at: " << start_time << std::endl;

    absl::Duration timeout = _timeout - absl::Seconds(1);

    const int number_of_cores = 6;
    TotalCost node_global_best_cost[number_of_cores];
    std::optional<Solution> node_global_best_solution[number_of_cores];

    ProblemInstance problem = convertToProblemInstance(_problem);
    std::cout << "# Convertion to problem instance finished: " << absl::ToDoubleSeconds(absl::Now() - start_time) << std::endl;
#pragma omp parallel for
    for (int core = 0; core < number_of_cores; ++core)
    {
      // We modify problem instance for simpler expression.
      TotalCost global_best_cost = std::numeric_limits<TotalCost>::max();
      std::optional<Solution> global_best_solution;
      const long long TICK = 1000000000000000000;

      // Initialize solution
      State state = create_initial_state(problem);
      while (absl::Now() - start_time < timeout)
      {
        TotalCost cur_best_cost = std::numeric_limits<TotalCost>::max();
        std::optional<Solution> cur_best_solution;

        absl::Time last_updated = absl::Now();
        absl::Time last_output = absl::Now() - absl::Seconds(10);

        int trial = 0;
        int accepted = 0;
        std::map<int, int> scores;

        while (absl::Now() - start_time < timeout && absl::Now() - last_updated < absl::Seconds(10))
        {

          auto current = absl::Now();
          if (current - last_output > absl::Seconds(1))
          {
            last_output = current;
            output("# Found solution with cost: " << cur_best_cost);
            output("# Time from start: " << absl::ToDoubleSeconds(current - start_time));
            output("# Trial/accepted: " << trial << "/" << accepted);
          }

          trial++;
          const int current_prob = xor64() % 1000;
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
          targets.reserve(vertex.selectables[selection].connection_costs.size() + 1);
          prevs.reserve(vertex.selectables[selection].connection_costs.size() + 1);
          targets.push_back(Operation{node, selection});
          prevs.push_back(Operation{node, old});

          if (vertex.selectables[selection].connection_costs.size() >= 2)
          {
            const auto &connection_costs = vertex.selectables[selection].connection_costs;
            for (const auto &x : connection_costs)
            {
              int target_node = x.first;
              std::vector<int> cost_oks;
              bool no_fix = false;
              int priority = 100;
              const auto &costs = x.second;
              const size_t costs_size = costs.size();
              for (int j = 0; j < costs_size; ++j)
              {
                int go = costs[j] / (TICK / 10);
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
              const auto &target_vertex = problem.vertexs[target_node];
              const auto &target_selectables = target_vertex.selectables;
              const Usage min_usage = target_selectables[target_selection].usage;
              for (int j = 1; j < cost_oks.size(); ++j)
              {
                if (target_selectables[cost_oks[j]].usage < min_usage)
                {
                  target_selection = cost_oks[j];
                }
              }
              targets.push_back(Operation{target_node, target_selection});
              prevs.push_back(Operation{target_node, state.solution[target_node]});
            }
          }
          TotalCost old_cost = state.get_cost();
          scores[targets.size()]++;
          state.update(targets);
          if (!state.isValid)
          {
            state.update(prevs);
            continue;
          }
          TotalCost new_cost = state.get_cost();
          TotalCost cost_diff = new_cost - old_cost;

          if (cost_diff > 0)
          {
            if (((cost_diff >= state.current_cost / 100000 || xor64() % 1000 <= 990) && (xor64() % 1000000 != 0)))
            {
              // revert.
              state.update(prevs);
              continue;
            }
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
              output("# Found solution with cost: " << cur_best_cost);
              output("# Time from start: " << absl::ToDoubleSeconds(current - start_time));
              output("# Trial/accepted: " << trial << "/" << accepted);
            }
          }
          accepted++;
        }
        TotalCost single_cost = 0;
        for (int i = 0; i < cur_best_solution->size(); i++)
        {
          single_cost += problem.vertexs[i].selectables[(*cur_best_solution)[i]].single_cost;
        }
        output("# Single cost: " << single_cost);
        output("# connection_cost: " << cur_best_cost - single_cost);
        output("# Final cost: " << cur_best_cost);
        output("# Time from start: " << absl::ToDoubleSeconds(absl::Now() - start_time));
        output("# Trial/accepted: " << trial << "/" << accepted);
        for (auto x : scores)
        {
          output("# " << x.first << " : " << x.second);
        }
        if (global_best_cost > state.current_cost)
        {
          global_best_cost = state.current_cost;
          global_best_solution = cur_best_solution;
        }
      }
      node_global_best_cost[core] = global_best_cost;
      node_global_best_solution[core] = global_best_solution;
    }
    TotalCost the_best = std::numeric_limits<TotalCost>::max();
    std::optional<Solution> the_best_solution;
    for (int i = 0; i < number_of_cores; ++i)
    {
      std::cout << "# core: " << i << " final_cost: " << node_global_best_cost[i] << std::endl;
      if (the_best > node_global_best_cost[i])
      {
        the_best = node_global_best_cost[i];
        the_best_solution = node_global_best_solution[i];
      }
    }
    std::cout << "# Final cost: " << the_best << std::endl;

    /*
    // validate solution.
    absl::StatusOr<TotalCost> verify_result = Evaluate(_problem, the_best_solution.value());
    if (verify_result.ok() == false || verify_result.value() != the_best)
    {
      std::cerr << "# Verification failed." << std::endl;
      return absl::InvalidArgumentError("Verification failed.");
    }
    */
    absl::Time current_time = absl::Now();
    std::cout << "# Exiting Solver::solve at: " << current_time << std::endl;
    std::cout << "# Elapsed: " << absl::ToDoubleSeconds(current_time - start_time) << std::endl;
    return *the_best_solution;
  }

} // namespace iopddl
