#pragma once

// sg::exec: Coroutines.

#include "../core/graph.hpp"
#include "sync_wait.hpp"

#include <concepts>
#include <coroutine>
#include <exception>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace sg::exec {

namespace detail {
// Suspends the coroutine, then calls fn, which may destroy it.
template <class Fn> struct suspend_then {
  Fn fn;
  bool await_ready() const noexcept { return false; }
  void await_suspend(std::coroutine_handle<>) noexcept { fn(); }
  [[noreturn]] void await_resume() noexcept { std::terminate(); }
};
template <class Fn> suspend_then(Fn) -> suspend_then<Fn>;

// The operation of an awaitable used as a sender: a coroutine, started by
// start(), that completes the receiver from a suspended state.
template <class R> class awaitable_operation {
public:
  struct promise_type : with_await_transform<promise_type> {
    R &rcvr;
    template <class A> promise_type(A &, R &r) noexcept : rcvr(r) {}
    awaitable_operation get_return_object() noexcept {
      return awaitable_operation{
        std::coroutine_handle<promise_type>::from_promise(*this)};
    }
    std::suspend_always initial_suspend() noexcept { return {}; }
    [[noreturn]] std::suspend_always final_suspend() noexcept {
      std::terminate();
    }
    [[noreturn]] void unhandled_exception() noexcept { std::terminate(); }
    [[noreturn]] void return_void() noexcept { std::terminate(); }
    std::coroutine_handle<> unhandled_stopped() noexcept {
      exec::set_stopped(std::move(rcvr));
      return std::noop_coroutine();
    }
    decltype(auto) get_env() const noexcept { return exec::get_env(rcvr); }
  };

  using operation_state_concept = operation_state_t;
  awaitable_operation(awaitable_operation &&other) noexcept
      : handle_(std::exchange(other.handle_, {})) {}
  ~awaitable_operation() {
    if (handle_)
      handle_.destroy();
  }
  void start() & noexcept { handle_.resume(); }

private:
  explicit awaitable_operation(std::coroutine_handle<promise_type> h) noexcept
      : handle_(h) {}
  std::coroutine_handle<promise_type> handle_;
};

template <class A, class R> awaitable_operation<R> connect_awaitable(A a, R r) {
  using result =
    await_result_t<A, typename awaitable_operation<R>::promise_type>;
  std::exception_ptr ep;
  try {
    if constexpr (std::is_void_v<result>) {
      co_await std::move(a);
      co_await suspend_then{[&]() noexcept { exec::set_value(std::move(r)); }};
    } else {
      auto &&v = co_await std::move(a);
      co_await suspend_then{[&]() noexcept {
        exec::set_value(std::move(r), std::forward<decltype(v)>(v));
      }};
    }
  } catch (...) {
    ep = std::current_exception();
  }
  co_await suspend_then{
    [&]() noexcept { exec::set_error(std::move(r), std::move(ep)); }};
}

// What co_await of a sender produces from its value types: void for none or
// an empty one, the value for one, a tuple for several.
template <class Values> struct single_value {};
template <> struct single_value<type_list<>> {
  using type = void;
};
template <> struct single_value<type_list<std::tuple<>>> {
  using type = void;
};
template <class V> struct single_value<type_list<std::tuple<V>>> {
  using type = V;
};
template <class... Vs> struct single_value<type_list<std::tuple<Vs...>>> {
  using type = std::tuple<Vs...>;
};

template <class S, class Promise>
using await_values_t = value_types_of_t<S, std::decay_t<env_of_t<Promise &>>,
  decayed_tuple, type_list>;

template <class S, class Promise>
concept awaitable_sender = sender_in<S, std::decay_t<env_of_t<Promise &>>> &&
  requires { typename single_value<await_values_t<S, Promise>>::type; } &&
  requires(Promise &p) {
    { p.unhandled_stopped() } -> std::convertible_to<std::coroutine_handle<>>;
  };

// The awaiter whose start() is running on this thread, and the flag that its
// await_suspend reads afterwards.
struct starting_awaiter {
  const void *awaiter = nullptr;
  bool *completed = nullptr;
};
inline thread_local starting_awaiter starting{};

// An operation that can tell which scheduler it completed on (a task's).
template <class Op, class Sch>
concept reports_scheduler = requires(const Op &op) {
  { op.current_scheduler() } -> std::same_as<const Sch &>;
};

// The awaiter of sender S in a coroutine whose promise is Promise. The
// operation lives in the awaiter, in the coroutine frame, so co_await does not
// allocate; the receiver's environment is the promise's. A completion on this
// thread during start() only sets a flag on await_suspend's stack, which then
// lets the coroutine continue inline, so the stack does not grow however many
// such senders it awaits. Any other completion resumes the coroutine where it
// happens; await_suspend does not touch the awaiter after start(), so that may
// be before it returns. With Affine, such a completion first schedules onto
// the promise's scheduler, unless the sender is known to complete there.
template <class S, class Promise, bool Affine> class sender_awaiter {
  using env_type = std::decay_t<env_of_t<Promise &>>;
  static_assert(type_list_size<await_values_t<S, Promise>>::value <= 1,
    "co_await requires a sender with at most one value completion; use "
    "into_variant");
  using value_type = typename single_value<await_values_t<S, Promise>>::type;
  using stored_type =
    std::conditional_t<std::is_void_v<value_type>, std::monostate, value_type>;

  struct receiver {
    using receiver_concept = receiver_t;
    sender_awaiter *self;
    template <class... Vs> void set_value(Vs &&...vs) && noexcept {
      try {
        self->result.template emplace<1>(std::forward<Vs>(vs)...);
      } catch (...) {
        self->result.template emplace<2>(std::current_exception());
      }
      self->arrive();
    }
    template <class E> void set_error(E &&e) && noexcept {
      self->result.template emplace<2>(as_exception_ptr(std::forward<E>(e)));
      self->arrive();
    }
    void set_stopped() && noexcept { self->arrive(); }
    env_type get_env() const noexcept { return exec::get_env(self->promise); }
  };

  struct no_hop {};
  // Scheduling back onto the promise's scheduler. Only used with Affine.
  struct hop {
    using scheduler_type =
      std::decay_t<decltype(get_scheduler(std::declval<const env_type &>()))>;
    struct resume_receiver {
      using receiver_concept = receiver_t;
      sender_awaiter *self;
      void set_value() && noexcept { self->handle.resume(); }
      template <class E> void set_error(E &&e) && noexcept {
        self->result.template emplace<2>(as_exception_ptr(std::forward<E>(e)));
        self->handle.resume();
      }
      void set_stopped() && noexcept {
        self->promise.unhandled_stopped().resume();
      }
      env_type get_env() const noexcept { return exec::get_env(self->promise); }
    };
    struct operation {
      scheduler_type sch;
      connect_result_t<schedule_result_t<scheduler_type &>, resume_receiver> op;
      operation(scheduler_type s, resume_receiver r)
          : sch(std::move(s)), op(exec::connect(exec::schedule(sch), r)) {}
    };
    std::optional<operation> op;
  };
  using op_type = connect_result_t<S, receiver>;

  Promise &promise;
  std::coroutine_handle<Promise> handle;
  // Still monostate after the completion: stopped.
  std::variant<std::monostate, stored_type, std::exception_ptr> result;
  bool sends_on_scheduler; // S reports the promise's scheduler
  [[no_unique_address]] std::conditional_t<Affine, hop, no_hop> back;
  op_type op;

  static bool completes_on(
    [[maybe_unused]] const S &s, [[maybe_unused]] const Promise &p) noexcept {
    if constexpr (Affine) {
      using sch_type = typename hop::scheduler_type;
      if constexpr (requires {
                      {
                        get_completion_scheduler<set_value_t>(exec::get_env(s))
                      } -> std::same_as<sch_type>;
                    })
        return get_completion_scheduler<set_value_t>(exec::get_env(s)) ==
          get_scheduler(exec::get_env(p));
    }
    return false;
  }

  bool on_scheduler() const noexcept {
    if constexpr (reports_scheduler<op_type, typename hop::scheduler_type>)
      if (op.current_scheduler() == get_scheduler(exec::get_env(promise)))
        return true;
    return sends_on_scheduler;
  }

  void arrive() noexcept {
    if (starting.awaiter == this)
      *starting.completed = true;
    else
      resume();
  }

  // Continues the coroutine after a completion outside start().
  void resume() noexcept {
    if (result.index() == 0) {
      promise.unhandled_stopped().resume();
      return;
    }
    if constexpr (Affine) {
      if (!on_scheduler()) {
        try {
          back.op.emplace(get_scheduler(exec::get_env(promise)),
            typename hop::resume_receiver{this});
        } catch (...) {
          result.template emplace<2>(std::current_exception());
          handle.resume();
          return;
        }
        exec::start(back.op->op);
        return;
      }
    }
    handle.resume();
  }

public:
  sender_awaiter(S &&s, Promise &p)
      : promise(p), sends_on_scheduler(completes_on(s, p)),
        op(exec::connect(std::forward<S>(s), receiver{this})) {}
  sender_awaiter(sender_awaiter &&) = delete;

  bool await_ready() const noexcept { return false; }
  bool await_suspend(std::coroutine_handle<Promise> h) noexcept {
    handle = h;
    bool completed = false;
    starting_awaiter outer = std::exchange(starting, {this, &completed});
    exec::start(op);
    starting = outer;
    if (!completed)
      return true; // The completion resumes the coroutine, maybe already has.
    if (result.index() == 0) {
      promise.unhandled_stopped().resume();
      return true;
    }
    return false;
  }
  value_type await_resume() {
    if (result.index() == 2)
      std::rethrow_exception(std::get<2>(result));
    if constexpr (!std::is_void_v<value_type>)
      return std::move(std::get<1>(result));
  }
};

template <bool Affine, class S, class Promise>
auto await_sender(S &&s, Promise &p) {
  return sender_awaiter<S, Promise, Affine>(std::forward<S>(s), p);
}
} // namespace detail

// What co_await e awaits in a coroutine with promise p: e.as_awaitable(p), an
// awaitable as is, or a sender with at most one value completion through an
// awaiter; set_error rethrows the error as an exception, and set_stopped calls
// p.unhandled_stopped().
struct as_awaitable_t {
  template <class Expr, class Promise>
  decltype(auto) operator()(Expr &&e, Promise &p) const {
    if constexpr (requires { std::forward<Expr>(e).as_awaitable(p); })
      return std::forward<Expr>(e).as_awaitable(p);
    else if constexpr (detail::is_awaitable<Expr, detail::plain_promise>)
      return std::forward<Expr>(e);
    else if constexpr (detail::awaitable_sender<Expr, Promise>)
      return detail::await_sender<false>(std::forward<Expr>(e), p);
    else
      return std::forward<Expr>(e);
  }
};
inline constexpr as_awaitable_t as_awaitable{};

// A base for promise types: co_await works on senders, and a stopped sender
// stops the awaiting coroutine (see set_continuation).
template <class Promise> class with_awaitable_senders {
  [[noreturn]] static std::coroutine_handle<> no_stopped_handler(
    void *) noexcept {
    std::terminate();
  }
  std::coroutine_handle<> continuation_{};
  std::coroutine_handle<> (*stopped_)(void *) noexcept = &no_stopped_handler;

public:
  // h is resumed when this coroutine finishes; a stop goes to h's promise.
  template <class OtherPromise>
    requires(!std::is_void_v<OtherPromise>)
  void set_continuation(std::coroutine_handle<OtherPromise> h) noexcept {
    continuation_ = h;
    if constexpr (requires(OtherPromise &other) { other.unhandled_stopped(); })
      stopped_ = [](void *address) noexcept -> std::coroutine_handle<> {
        return std::coroutine_handle<OtherPromise>::from_address(address)
          .promise()
          .unhandled_stopped();
      };
    else
      stopped_ = &no_stopped_handler;
  }
  std::coroutine_handle<> continuation() const noexcept {
    return continuation_;
  }
  std::coroutine_handle<> unhandled_stopped() noexcept {
    return stopped_(continuation_.address());
  }
  template <class Value> decltype(auto) await_transform(Value &&value) {
    return as_awaitable(
      std::forward<Value>(value), static_cast<Promise &>(*this));
  }
};

} // namespace sg::exec
