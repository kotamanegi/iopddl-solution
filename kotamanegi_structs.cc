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

#include "iopddl.h"
#include "kotamanegi_structs.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "nlohmann/json.hpp"

////////////////////////////////////////////////////////////////////////////////
/////////  Utilities for reading problems and evaluating solutions.    /////////
/////////  Contest participants do not need to modify this code.       /////////
////////////////////////////////////////////////////////////////////////////////

namespace iopddl
{
  ProblemInstance convertToProblemInstance(const Problem &problem)
  {
    // simple conversion
    ProblemInstance instance;
    instance.name = problem.name;
    instance.usage_limit = problem.usage_limit;
    for (const Node &node : problem.nodes)
    {
      Vertex vertex;
      vertex.interval = node.interval;
      for (const Strategy &strategy : node.strategies)
      {
        Selection selection;
        selection.usage = strategy.usage;
        selection.single_cost = strategy.cost;
        vertex.selectables.push_back(selection);
      }
      instance.vertexs.push_back(vertex);
    }
    for (const Edge &edge : problem.edges)
    {
      Vertex &vertex0 = instance.vertexs[edge.nodes[0]];
      Vertex &vertex1 = instance.vertexs[edge.nodes[1]];

      for (int i = 0; i < vertex0.selectables.size(); i++)
      {
        std::vector<Cost> costs;
        for (int j = 0; j < vertex1.selectables.size(); j++)
        {
          costs.push_back(edge.strategies[i * vertex1.selectables.size() + j].cost);
        }
        std::pair<VertexIdx, std::vector<Cost>> connection_cost = {edge.nodes[1], costs};
        vertex0.selectables[i].connection_costs.push_back(connection_cost);
      }

      for (int i = 0; i < vertex1.selectables.size(); i++)
      {
        std::vector<Cost> costs;
        for (int j = 0; j < vertex0.selectables.size(); j++)
        {
          costs.push_back(edge.strategies[j * vertex1.selectables.size() + i].cost);
        }
        std::pair<VertexIdx, std::vector<Cost>> connection_cost = {edge.nodes[0], costs};
        vertex1.selectables[i].connection_costs.push_back(connection_cost);
      }
    }
    return zipTimeInterval(instance);
  }
  ProblemInstance zipTimeInterval(const ProblemInstance &problem)
  {
    std::map<TimeIdx,int> time_map;
    for (const Vertex &vertex : problem.vertexs)
    {
      time_map[vertex.interval.first] = 0;
      time_map[vertex.interval.second] = 0;
    }
    int time_idx = 0;
    for(auto &time : time_map){
      time.second = time_idx++;
    }
    ProblemInstance instance;
    instance.name = problem.name;
    instance.usage_limit = problem.usage_limit;
    for (const Vertex &vertex : problem.vertexs)
    {
      Vertex new_vertex;
      new_vertex.interval = {time_map[vertex.interval.first], time_map[vertex.interval.second]};
      new_vertex.selectables = vertex.selectables;
      instance.vertexs.push_back(new_vertex);
    }
    return instance;
  }
  absl::StatusOr<TotalCost> FastEvaluate(const ProblemInstance &problem,
                                         const Solution &solution)
  {
    if (solution.size() != problem.vertexs.size()) {
      return absl::InvalidArgumentError("Incorrect solution size");
    }
    
    TotalCost cost = 0;
    // use double-cost for each selection
    for (int i = 0; i < solution.size(); i++)
    {
      const Vertex &vertex = problem.vertexs[i];
      const Selection &selection = vertex.selectables[solution[i]];
      cost += selection.single_cost * 2;
      for (int j = 0; j < selection.connection_costs.size(); j++)
      {
        int targetVertexIdx = selection.connection_costs[j].first;
        cost += selection.connection_costs[j].second[solution[targetVertexIdx]];
      }
    }
    cost /= 2;

    if(problem.usage_limit){
      TimeIdx max_time = 0;
      for (const Vertex &vertex : problem.vertexs)
      {
        max_time = std::max(max_time, vertex.interval.second);
      }
      std::vector<TotalUsage> total_usages(max_time + 1);
      for(int i = 0;i < solution.size();++i){
        const Vertex &vertex = problem.vertexs[i];
        const Selection &selection = vertex.selectables[solution[i]];
        total_usages[vertex.interval.first] += selection.usage;
        total_usages[vertex.interval.second] -= selection.usage;
      }
      for(int i = 1;i <= max_time;++i){
        total_usages[i] += total_usages[i - 1];
        if(total_usages[i] > *problem.usage_limit){
          return absl::ResourceExhaustedError("Usage limit exceeded");
        }
      }
    }
    return cost;
  }
} // namespace iopddl
