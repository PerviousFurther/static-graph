// Async scopes with only their own headers: spawn, spawn_future and join.
#include <sg/exec/just.hpp>
#include <sg/exec/scope.hpp>
#include <sg/exec/sync_wait.hpp>
#include <sg/exec/then.hpp>

namespace ex = sg::exec;

int main() {
  ex::counting_scope s;
  int n = 0;
  ex::spawn(ex::just() | ex::then([&]() noexcept { ++n; }), s.get_token());
  auto f = ex::spawn_future(ex::just(2), s.get_token());
  auto [v] = *ex::sync_wait(std::move(f));
  s.close();
  ex::sync_wait(s.join());
  return n == 1 && v == 2 ? 0 : 1;
}
