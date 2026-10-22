// when_all, let_value, continues_on and stopped_as_optional with only their
// own headers.
#include <sg/exec/continues_on.hpp>
#include <sg/exec/schedulers.hpp>
#include <sg/exec/sync_wait.hpp>
#include <sg/exec/then.hpp>
#include <sg/exec/when_all.hpp>

namespace ex = sg::exec;

int main() {
  auto r = ex::sync_wait(
    ex::when_all(ex::just(1),
      ex::just(2) | ex::let_value([](int x) { return ex::just(x * 10); })) |
    ex::continues_on(ex::inline_scheduler{}) |
    ex::then([](int a, int b) { return a + b; }));
  auto o = ex::sync_wait(ex::just(5) | ex::stopped_as_optional());
  return std::get<0>(*r) == 21 && *std::get<0>(*o) == 5 ? 0 : 1;
}
