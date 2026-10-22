// task with only its own headers: awaiting a task and a sender.
#include <sg/exec/just.hpp>
#include <sg/exec/sync_wait.hpp>
#include <sg/exec/task.hpp>

namespace ex = sg::exec;

ex::task<int> one() { co_return 1; }
ex::task<int> two() {
  int x = co_await one();
  int y = co_await ex::just(1);
  co_return x + y;
}

int main() {
  auto [v] = *ex::sync_wait(two());
  return v == 2 ? 0 : 1;
}
