// Sender/receiver example on inc/exec.hpp.
// Build: g++ -std=c++20 -O2 -pthread -I inc tests/exec/exec.cpp -o exec

#include <sg/exec/exec.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <system_error>

namespace ex = sg::exec;

#define CHECK(...)                                                             \
  do {                                                                         \
    if (!(__VA_ARGS__)) {                                                      \
      std::fprintf(stderr, "FAILED line %d: %s\n", __LINE__, #__VA_ARGS__);    \
      std::abort();                                                            \
    }                                                                          \
  } while (false)

// notify while locked: wait may destroy the operation immediately on return.
template <class... Vs> struct result {
  std::mutex mutex;
  std::condition_variable ready;
  bool done = false;
  unsigned calls = 0;
  std::optional<std::tuple<Vs...>> values;
  std::exception_ptr error;
  bool failed = false;
  bool stopped = false;
  std::thread::id thread;

  template <class... Args> void set_value(Args &&...args) noexcept {
    finish([&] { values.emplace(std::forward<Args>(args)...); });
  }
  template <class E> void set_error(E &&e) noexcept {
    finish([&] {
      failed = true;
      if constexpr (std::is_same_v<std::decay_t<E>, std::exception_ptr>)
        error = e;
    });
  }
  void set_stopped() noexcept {
    finish([&] { stopped = true; });
  }

  template <class F> void finish(F f) noexcept {
    std::lock_guard lock(mutex);
    ++calls;
    thread = std::this_thread::get_id();
    try {
      f();
    } catch (...) {
      failed = true;
      error = std::current_exception();
    }
    done = true;
    ready.notify_all();
  }
  bool is_done() {
    std::lock_guard lock(mutex);
    return done;
  }
  void wait() {
    std::unique_lock lock(mutex);
    CHECK(ready.wait_for(lock, std::chrono::seconds(10), [&] { return done; }));
    CHECK(calls == 1);
  }
};

// A receiver that forwards to a result and optionally provides a scheduler.
template <class R, class Env = ex::env<>> struct receiver_ref {
  using receiver_concept = ex::receiver_t;
  R *target;
  Env environment;
  explicit receiver_ref(R *t, Env e = Env())
      : target(t), environment(std::move(e)) {}
  template <class... Args> void set_value(Args &&...args) && noexcept {
    target->set_value(std::forward<Args>(args)...);
  }
  template <class E> void set_error(E &&e) && noexcept {
    target->set_error(std::forward<E>(e));
  }
  void set_stopped() && noexcept { target->set_stopped(); }
  const Env &get_env() const noexcept { return environment; }
};

struct worker {
  ex::run_loop loop;
  std::thread thread{[this] { loop.run(); }};
  ~worker() {
    loop.finish();
    thread.join();
  }
  auto scheduler() noexcept { return loop.get_scheduler(); }
};

// A sender written against the standard protocol, not built from descriptors.
struct outcome_sender {
  using sender_concept = ex::sender_t;
  using completion_signatures = ex::completion_signatures<ex::set_value_t(int),
    ex::set_error_t(std::error_code), ex::set_stopped_t()>;
  int mode;
  template <class R> struct operation {
    int mode;
    R receiver;
    void start() & noexcept {
      if (mode == 0)
        ex::set_value(std::move(receiver), 1);
      else if (mode == 1)
        ex::set_error(
          std::move(receiver), std::make_error_code(std::errc::io_error));
      else
        ex::set_stopped(std::move(receiver));
    }
  };
  template <class R> operation<R> connect(R r) && {
    return {mode, std::move(r)};
  }
};

template <class S, class R> void run(S s, R &r) {
  auto op = ex::connect(std::move(s), receiver_ref<R>{&r});
  ex::start(op);
  r.wait();
}

void basics() {
  result<int> r;
  auto source = ex::just(2) | ex::then([](int x) { return x + 1; });
  static_assert(ex::sender<decltype(source)>);
  static_assert(std::is_same_v<ex::completion_signatures_of_t<decltype(source)>,
    ex::completion_signatures<ex::set_value_t(int),
      ex::set_error_t(std::exception_ptr)>>);
  auto op = ex::connect(std::move(source), receiver_ref<result<int>>{&r});
  static_assert(!std::is_move_constructible_v<decltype(op)>);
  static_assert(decltype(op)::node_count == 3);
  CHECK(!r.is_done());
  CHECK(std::get<0>(op.get_node<2>()->values) == 2);
  ex::start(op);
  r.wait();
  CHECK(std::get<0>(*r.values) == 3);

  auto [v] = ex::sync_wait(ex::then(ex::just(20), [](int x) noexcept {
    return x * 2;
  })).value();
  CHECK(v == 40);
  auto recovered =
    ex::sync_wait(ex::just(1) | ex::then([](int) -> int { throw 42; }) |
      ex::let_error([](std::exception_ptr &) { return ex::just(42); }));
  CHECK(std::get<0>(*recovered) == 42);
  auto from_stop = ex::sync_wait(
    ex::just_stopped() | ex::let_stopped([] { return ex::just(7); }));
  CHECK(std::get<0>(*from_stop) == 7);
  CHECK(!ex::sync_wait(outcome_sender{2}).has_value());
  auto upon = ex::sync_wait(outcome_sender{1} |
    ex::upon_error([](std::error_code ec) noexcept { return ec.value(); }));
  CHECK(std::get<0>(*upon) == int(std::errc::io_error));
  bool threw = false;
  try {
    ex::sync_wait(
      outcome_sender{1} | ex::then([](int x) noexcept { return x; }));
  } catch (const std::system_error &e) {
    threw = e.code() == std::errc::io_error;
  }
  CHECK(threw);
  auto joined = ex::sync_wait(ex::when_all(outcome_sender{0}, ex::just(2)));
  CHECK(*joined == std::tuple{1, 2});
  std::puts("PASS just/then/upon/let, signatures, sync_wait, foreign sender, "
            "error_code");
}

void multiple_signatures() {
  // let_value may pick different senders per value signature.
  auto choose = ex::just(3) | ex::let_value([](int x) {
    return ex::just(x) | ex::then([](int y) { return std::to_string(y); });
  });
  CHECK(std::get<0>(*ex::sync_wait(std::move(choose))) == "3");

  auto opt = ex::sync_wait(ex::just(5) | ex::stopped_as_optional());
  CHECK(std::get<0>(*opt) == std::optional<int>(5));
  auto none = ex::sync_wait(ex::just_stopped() |
    ex::let_stopped([] { return ex::just_stopped(); }) |
    ex::upon_stopped([]() noexcept { return 0; }));
  CHECK(std::get<0>(*none) == 0);
  auto empty = ex::sync_wait(outcome_sender{2} | ex::stopped_as_optional());
  CHECK(!std::get<0>(*empty).has_value());

  result<int> as_error;
  run(
    ex::just_stopped() | ex::stopped_as_error(17) | ex::then([] { return 0; }),
    as_error);
  CHECK(as_error.failed);

  auto variant = ex::sync_wait_with_variant(
    ex::just(1) | ex::let_value([](int x) { return ex::just(x, 2.5); }));
  auto &alt = std::get<std::tuple<int, double>>(std::get<0>(*variant));
  CHECK(std::get<0>(alt) == 1 && std::get<1>(alt) == 2.5);
  std::puts(
    "PASS multiple signatures, into_variant, stopped_as_optional/error");
}

void environments(worker &a) {
  ex::inplace_stop_source source;
  auto token = ex::sync_wait(ex::write_env(ex::read_env(ex::get_stop_token),
    ex::prop{ex::get_stop_token, source.get_token()}));
  CHECK(std::get<0>(*token) == source.get_token());
  auto sched = ex::sync_wait(ex::starts_on(a.scheduler(),
    ex::read_env(ex::get_scheduler) |
      ex::then([](auto s) noexcept { return s; })));
  CHECK(std::get<0>(*sched) == a.scheduler());
  auto completion = ex::get_completion_scheduler<ex::set_value_t>(
    ex::get_env(ex::schedule(a.scheduler())));
  CHECK(completion == a.scheduler());
  static_assert(ex::scheduler<decltype(a.scheduler())>);
  static_assert(ex::unstoppable_token<ex::never_stop_token>);
  static_assert(ex::stoppable_token<ex::inplace_stop_token> &&
    !ex::unstoppable_token<ex::inplace_stop_token>);
  static_assert(
    ex::sender_to<decltype(ex::just(1)), receiver_ref<result<int>>>);
  struct int_receiver {
    using receiver_concept = ex::receiver_t;
    void set_value(int) && noexcept {}
  };
  static_assert(ex::receiver_of<int_receiver,
    ex::completion_signatures<ex::set_value_t(int)>>);
  static_assert(
    !ex::receiver_of<int_receiver,
      ex::completion_signatures<ex::set_value_t(int), ex::set_stopped_t()>>);
  std::puts("PASS read_env/write_env, get_scheduler, completion scheduler");
}

void switching(worker &a, worker &b) {
  result<int> r;
  std::thread::id before, after;
  run(ex::schedule(a.scheduler()) | ex::then([&] {
    before = std::this_thread::get_id();
    return 9;
  }) | ex::continues_on(b.scheduler()) |
      ex::then([&](int x) {
        after = std::this_thread::get_id();
        return x;
      }),
    r);
  CHECK(before == a.thread.get_id());
  CHECK(after == b.thread.get_id());
  CHECK(r.thread == b.thread.get_id());
  CHECK(std::get<0>(*r.values) == 9);

  // on() returns to the receiver's scheduler.
  result<int> returned;
  auto env = ex::env{ex::prop{ex::get_scheduler, b.scheduler()}};
  auto op =
    ex::connect(ex::on(a.scheduler(), ex::just(5) | ex::then([&](int n) {
      CHECK(std::this_thread::get_id() == a.thread.get_id());
      return n;
    })),
      receiver_ref<result<int>, decltype(env)>{&returned, env});
  ex::start(op);
  returned.wait();
  CHECK(returned.thread == b.thread.get_id());

  // on(sndr, sch, closure) runs only the closure on sch.
  auto hopped =
    ex::sync_wait(ex::just(1) | ex::on(a.scheduler(), ex::then([&](int x) {
      CHECK(std::this_thread::get_id() == a.thread.get_id());
      return x + 1;
    })));
  CHECK(std::get<0>(*hopped) == 2);

  result<> error;
  run(ex::just_error(std::make_exception_ptr(13)) |
      ex::continues_on(b.scheduler()),
    error);
  CHECK(error.error && error.thread == b.thread.get_id());
  result<> stopped;
  run(ex::just_stopped() | ex::continues_on(a.scheduler()), stopped);
  CHECK(stopped.stopped && stopped.thread == a.thread.get_id());
  std::puts(
    "PASS schedule, continues_on, starts_on/on, value/error/stopped handoff");
}

void asynchronous_let(worker &a, worker &b) {
  result<int> r;
  run(ex::just(std::make_unique<int>(3)) |
      ex::let_value([&](std::unique_ptr<int> &input) {
        return ex::schedule(a.scheduler()) |
          ex::then([&input] { return *input + 1; }) |
          ex::let_value([&](int &x) {
            return ex::schedule(b.scheduler()) |
              ex::then([&x] { return x * 10; });
          });
      }),
    r);
  CHECK(std::get<0>(*r.values) == 40);
  CHECK(r.thread == b.thread.get_id());
  std::puts("PASS asynchronous let keeps inputs and child graphs alive");
}

void joins(worker &a, worker &b) {
  CHECK(ex::sync_wait(ex::when_all()).has_value());
  for (int i = 0; i < 200; ++i) {
    result<int, std::unique_ptr<int>, int> r;
    run(ex::when_all(ex::schedule(a.scheduler()) | ex::then([i] { return i; }),
          ex::just(std::make_unique<int>(17)),
          ex::schedule(b.scheduler()) | ex::then([] { return 23; })),
      r);
    CHECK(std::get<0>(*r.values) == i);
    CHECK(*std::get<1>(*r.values) == 17);
    CHECK(std::get<2>(*r.values) == 23);
  }
  auto nested =
    ex::sync_wait(ex::when_all(ex::when_all(ex::just(1)), ex::just(2)));
  CHECK(*nested == std::tuple{1, 2});
  std::puts("PASS empty/nested when_all, mixed sync/async, move-only values "
            "(200 runs)");
}

void cancellation(worker &a) {
  // An error stops the sibling that is still waiting in a queue.
  ex::run_loop delayed;
  result<int, int> r;
  bool ran = false;
  auto op = ex::connect(
    ex::when_all(ex::just(0) | ex::then([](int) -> int { throw 11; }),
      ex::schedule(delayed.get_scheduler()) | ex::then([&] {
        ran = true;
        return 2;
      })),
    receiver_ref<result<int, int>>{&r});
  ex::start(op);
  CHECK(!r.is_done());
  delayed.finish();
  delayed.run(); // The queued sibling sees the stop request.
  r.wait();
  CHECK(!ran && r.error);

  // A stop request from the receiver's environment reaches the children.
  ex::inplace_stop_source source;
  ex::run_loop held;
  result<int> outer;
  auto env = ex::env{ex::prop{ex::get_stop_token, source.get_token()}};
  auto stoppable = ex::connect(ex::when_all(ex::schedule(held.get_scheduler()) |
                                 ex::then([] { return 1; })),
    receiver_ref<result<int>, decltype(env)>{&outer, env});
  ex::start(stoppable);
  source.request_stop();
  held.finish();
  held.run();
  outer.wait();
  CHECK(outer.stopped);

  ex::inplace_stop_source early;
  early.request_stop();
  result<> pre;
  auto pre_env = ex::env{ex::prop{ex::get_stop_token, early.get_token()}};
  auto pre_op = ex::connect(ex::when_all(ex::schedule(a.scheduler())),
    receiver_ref<result<>, decltype(pre_env)>{&pre, pre_env});
  ex::start(pre_op);
  pre.wait();
  CHECK(pre.stopped);

  result<> stopped;
  run(ex::when_all(ex::just_stopped(), ex::just()), stopped);
  CHECK(stopped.stopped);
  result<> priority;
  run(ex::when_all(
        ex::just_stopped(), ex::just_error(std::make_exception_ptr(1))),
    priority);
  CHECK(priority.error && !priority.stopped);
  result<> closed;
  run(ex::schedule(delayed.get_scheduler()), closed);
  CHECK(closed.error);

  // Callbacks registered after a stop run immediately.
  ex::inplace_stop_source s;
  int hits = 0;
  {
    ex::inplace_stop_callback cb(s.get_token(), [&] { ++hits; });
    CHECK(s.request_stop() && !s.request_stop());
    ex::inplace_stop_callback late(s.get_token(), [&] { ++hits; });
  }
  CHECK(hits == 2);
  std::puts("PASS stop tokens, when_all cancellation, stopped/error priority");
}

struct throwing_move {
  bool *fail;
  explicit throwing_move(bool &flag) : fail(&flag) {}
  throwing_move(throwing_move &&other) : fail(other.fail) {
    if (*fail)
      throw 99;
  }
};

void storage_failures(worker &target) {
  bool fail = false;
  result<throwing_move> moved;
  auto transfer = ex::connect(
    ex::just(throwing_move{fail}) | ex::continues_on(target.scheduler()),
    receiver_ref<result<throwing_move>>{&moved});
  fail = true;
  ex::start(transfer);
  moved.wait();
  CHECK(moved.failed && moved.thread == std::this_thread::get_id());

  fail = false;
  result<throwing_move, int> joined;
  auto join =
    ex::connect(ex::when_all(ex::just(throwing_move{fail}), ex::just(1)),
      receiver_ref<result<throwing_move, int>>{&joined});
  fail = true;
  ex::start(join);
  joined.wait();
  CHECK(joined.failed);

  result<> null_error;
  run(ex::when_all(ex::just_error(std::exception_ptr{})), null_error);
  CHECK(null_error.failed && !null_error.error && !null_error.values);

  int value = 8;
  auto decayed = ex::sync_wait(ex::when_all(ex::when_all(ex::just() |
                                 ex::then([&]() -> int & { return value; }))) |
    ex::then([](int &&v) { return v; }));
  CHECK(std::get<0>(*decayed) == 8);

  ex::run_loop closed;
  closed.finish();
  result<int> rejected;
  run(ex::just(2) | ex::continues_on(closed.get_scheduler()), rejected);
  CHECK(rejected.failed);
  std::puts("PASS buffer exceptions, null error, decayed join values, rejected "
            "handoff");
}

struct deleting_receiver {
  using receiver_concept = ex::receiver_t;
  std::function<void()> *destroy;
  result<int> *output;
  // Consume graph-owned arguments before destroying their owner.
  void set_value(int value) && noexcept {
    auto *target = output;
    (*destroy)();
    target->set_value(value);
  }
  template <class E> void set_error(E &&) && noexcept {
    auto *target = output;
    (*destroy)();
    target->set_error(0);
  }
  void set_stopped() && noexcept {
    auto *target = output;
    (*destroy)();
    target->set_stopped();
  }
};

template <class S> void delete_on_completion(S s) {
  result<int> output;
  std::function<void()> destroy;
  using op_type = ex::connect_result_t<S, deleting_receiver>;
  auto op = std::unique_ptr<op_type>(new op_type(
    ex::connect(std::move(s), deleting_receiver{&destroy, &output})));
  destroy = [&] { op.reset(); };
  ex::start(*op);
  output.wait();
  CHECK(!op && output.values);
}

void lifetimes(worker &a, worker &b) {
  delete_on_completion(ex::just(1));
  delete_on_completion(ex::when_all(ex::just(1), ex::just(2)) |
    ex::then([](int x, int y) { return x + y; }));
  delete_on_completion(ex::just(1) | ex::continues_on(a.scheduler()));
  delete_on_completion(ex::just(1) | ex::let_value([&](int &v) {
    return ex::schedule(b.scheduler()) | ex::then([&v] { return v; });
  }));
  delete_on_completion(
    ex::when_all(ex::schedule(a.scheduler()) | ex::then([] { return 1; }),
      ex::schedule(b.scheduler()) | ex::then([] { return 2; })) |
    ex::then([](int x, int y) { return x + y; }));
  std::puts("PASS receiver destroys sync/async operation during completion");
}

void example(worker &a, worker &b, worker &ui) {
  auto [sum] = ex::sync_wait(
    ex::when_all(ex::schedule(a.scheduler()) | ex::then([] { return 20; }),
      ex::schedule(b.scheduler()) | ex::then([] { return 22; })) |
    ex::continues_on(ui.scheduler()) | ex::then([&](int x, int y) {
      CHECK(std::this_thread::get_id() == ui.thread.get_id());
      return x + y;
    })).value();
  std::printf("when_all -> continues_on(ui): %d\n", sum);
}

int main() {
  basics();
  multiple_signatures();
  worker a, b, ui;
  environments(a);
  switching(a, b);
  asynchronous_let(a, b);
  joins(a, b);
  cancellation(a);
  storage_failures(ui);
  lifetimes(a, b);
  example(a, b, ui);
  std::puts("All checks passed.");
}
