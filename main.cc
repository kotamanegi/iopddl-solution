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

#include <cstdlib>
#include <iostream>
#include <string>

#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "iopddl.h"
#include "solver.h"

int main(int argc, char *argv[])
{
  const std::string filename = argv[1];
  const absl::Duration timeout = absl::Seconds(atoi(argv[2]));
  const auto problem = iopddl::ReadProblem(filename);
  if (!problem.ok())
    exit(1);
  const auto solution = iopddl::Solver().Solve(*problem, timeout);
  const auto result = solution.value_or(iopddl::Solution{});
  std::cout << "[" << absl::StrJoin(result, ", ") << "]" << std::endl;
  return 0;
}
