// bulk with only its own headers: instantiates the adaptor end to end.
#include <sg/exec/bulk.hpp>
#include <sg/exec/just.hpp>
#include <sg/exec/sync_wait.hpp>

#include <execution>
#include <vector>

namespace ex = sg::exec;

int main() {
  auto r = ex::sync_wait(ex::just(std::vector<int>(4)) |
    ex::bulk(std::execution::seq, 4,
      [](int i, std::vector<int> &v) { v[i] = i; }));
  return std::get<0>(*r)[3] == 3 ? 0 : 1;
}
