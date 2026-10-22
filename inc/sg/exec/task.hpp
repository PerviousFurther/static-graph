#pragma once

// sg::exec: task.

#include "../core/graph.hpp"
#include "coroutine.hpp"
#include "task_scheduler.hpp"

#include <coroutine>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace sg::exec {

// co_yield with_error{e} completes the task with set_error(e).
template <class E> struct with_error {
  using type = std::remove_cvref_t<E>;
  type error;
};
template <class E> with_error(E) -> with_error<E>;

// co_await change_coroutine_scheduler{sch} moves the task onto sch, which it
// uses from then on, and returns the previous scheduler.
template <class Sch> struct change_coroutine_scheduler {
  using type = std::remove_cvref_t<Sch>;
  type scheduler;
};
template <class Sch>
change_coroutine_scheduler(Sch) -> change_coroutine_scheduler<Sch>;

namespace detail {
template <class T> struct is_change_scheduler : std::false_type {};
template <class Sch>
struct is_change_scheduler<change_coroutine_scheduler<Sch>> : std::true_type {};

// Env::name if Get<Env> names a type, else Default.
template <template <class> class Get, class Env, class Default>
struct member_type_or {
  using type = Default;
};
template <template <class> class Get, class Env, class Default>
  requires requires { typename Get<Env>; }
struct member_type_or<Get, Env, Default> {
  using type = Get<Env>;
};
template <class E> using scheduler_type_of = typename E::scheduler_type;
template <class E> using allocator_type_of = typename E::allocator_type;
template <class E> using stop_source_type_of = typename E::stop_source_type;
template <class E> using error_types_of = typename E::error_types;

// The allocator after the first allocator_arg among a coroutine's
// parameters, or a default one.
template <class Alloc> Alloc find_allocator() { return Alloc(); }
template <class Alloc, class A> Alloc find_allocator(const A &) {
  return Alloc();
}
template <class Alloc, class A, class B, class... Rest>
Alloc find_allocator(const A &, const B &b, const Rest &...rest) {
  if constexpr (std::is_same_v<A, std::allocator_arg_t>)
    return Alloc(b);
  else
    return find_allocator<Alloc>(b, rest...);
}

// Coroutine frames allocated with Alloc. A copy of the allocator is kept
// after the frame unless any default one can free it.
template <class Alloc> struct frame_allocation {
  struct alignas(__STDCPP_DEFAULT_NEW_ALIGNMENT__) block {
    std::byte bytes[__STDCPP_DEFAULT_NEW_ALIGNMENT__];
  };
  using block_alloc =
    typename std::allocator_traits<Alloc>::template rebind_alloc<block>;
  using traits = std::allocator_traits<block_alloc>;
  static constexpr bool stored = !(traits::is_always_equal::value &&
    std::is_default_constructible_v<block_alloc>);

  static std::size_t offset(std::size_t n) noexcept {
    constexpr std::size_t a = alignof(block_alloc);
    return (n + a - 1) / a * a;
  }
  static std::size_t blocks(std::size_t n) noexcept {
    std::size_t bytes = stored ? offset(n) + sizeof(block_alloc) : n;
    return (bytes + sizeof(block) - 1) / sizeof(block);
  }
  static void *allocate(std::size_t n, const Alloc &alloc) {
    block_alloc a(alloc);
    void *p = traits::allocate(a, blocks(n));
    if constexpr (stored)
      ::new (static_cast<std::byte *>(p) + offset(n)) block_alloc(std::move(a));
    return p;
  }
  static void deallocate(void *p, std::size_t n) noexcept {
    if constexpr (stored) {
      auto *kept = std::launder(reinterpret_cast<block_alloc *>(
        static_cast<std::byte *>(p) + offset(n)));
      block_alloc a(std::move(*kept));
      kept->~block_alloc();
      traits::deallocate(a, static_cast<block *>(p), blocks(n));
    } else {
      block_alloc a;
      traits::deallocate(a, static_cast<block *>(p), blocks(n));
    }
  }
};

// Where a task delivers its completion; the result is in its promise. Each
// returns the coroutine to resume next.
struct task_continuation {
  std::coroutine_handle<> (*complete)(task_continuation &) noexcept;
  std::coroutine_handle<> (*stopped)(task_continuation &) noexcept;
};

template <class T> struct task_value {
  T value;
};
template <> struct task_value<void> {};

// return_value or return_void, storing into result.
template <class T, class Result> struct task_result {
  Result result;
  template <class V = T> void return_value(V &&v) {
    result.template emplace<task_value<T>>(std::forward<V>(v));
  }
};
template <class Result> struct task_result<void, Result> {
  Result result;
  void return_void() noexcept { result.template emplace<task_value<void>>(); }
};
} // namespace detail

// A lazy coroutine sender. Environment may define scheduler_type (default
// task_scheduler), allocator_type (std::allocator<std::byte>),
// stop_source_type (inplace_stop_source) and error_types
// (completion_signatures<set_error_t(std::exception_ptr)>).
//
// Started, the task runs inline until its first suspension. It takes its
// scheduler from the receiver's environment, and after a co_await of a
// sender that completed later, it resumes on that scheduler; a sender that
// completed during start() resumes it inline. A stopped sender stops the
// task. Its environment answers get_scheduler, get_stop_token (the
// receiver's) and get_allocator (the one after std::allocator_arg among the
// coroutine's parameters).
template <class T = void, class Environment = env<>> class task {
public:
  using sender_concept = sender_t;
  using scheduler_type =
    typename detail::member_type_or<detail::scheduler_type_of, Environment,
      task_scheduler>::type;
  using allocator_type =
    typename detail::member_type_or<detail::allocator_type_of, Environment,
      std::allocator<std::byte>>::type;
  using stop_source_type =
    typename detail::member_type_or<detail::stop_source_type_of, Environment,
      inplace_stop_source>::type;
  using stop_token_type =
    decltype(std::declval<const stop_source_type &>().get_token());
  using error_types = typename detail::member_type_or<detail::error_types_of,
    Environment, type_list<set_error_t(std::exception_ptr)>>::type;
  using completion_signatures =
    detail::concat_sigs_t<type_list<typename detail::value_sig<T>::type>,
      error_types, type_list<set_stopped_t()>>;

  template <class R> class state;
  class promise_type;

  task(task &&other) noexcept : handle_(std::exchange(other.handle_, {})) {}
  ~task() {
    if (handle_)
      handle_.destroy();
  }

  template <receiver R> state<R> connect(R r) && {
    static_assert(receiver_of<R, completion_signatures>,
      "connect: the receiver does not accept every completion of the task");
    return state<R>(std::exchange(handle_, {}), std::move(r));
  }

private:
  template <class E>
  static constexpr bool has_error = std::is_same_v<
    detail::concat_sigs_t<error_types, type_list<set_error_t(E)>>, error_types>;
  template <class... Es>
  using result_variant =
    std::variant<std::monostate, detail::task_value<T>, Es...>;
  using result_type = detail::gather_signatures_t<set_error_t, error_types,
    detail::single_t, result_variant>;

  explicit task(std::coroutine_handle<promise_type> h) noexcept : handle_(h) {}
  std::coroutine_handle<promise_type> handle_;
};

template <class T, class Environment>
class task<T, Environment>::promise_type
    : public detail::task_result<T, result_type> {
  static constexpr bool affine =
    !std::is_same_v<scheduler_type, inline_scheduler>;
  using frames = detail::frame_allocation<allocator_type>;

  struct completion {
    bool await_ready() const noexcept { return false; }
    std::coroutine_handle<> await_suspend(
      std::coroutine_handle<promise_type> h) noexcept {
      auto &c = *h.promise().continuation_;
      return c.complete(c);
    }
    void await_resume() const noexcept {}
  };

  struct env_type {
    const promise_type *p;
    scheduler_type query(get_scheduler_t) const noexcept { return *p->sch_; }
    stop_token_type query(get_stop_token_t) const noexcept { return p->token_; }
    allocator_type query(get_allocator_t) const noexcept { return p->alloc_; }
  };

  detail::task_continuation *continuation_ = nullptr;
  std::optional<scheduler_type> sch_;
  stop_token_type token_;
  [[no_unique_address]] allocator_type alloc_;
  template <class> friend class state;

public:
  template <class... Args>
  promise_type(const Args &...args)
      : alloc_(detail::find_allocator<allocator_type>(args...)) {}

  template <class... Args>
  static void *operator new(std::size_t n, const Args &...args) {
    return frames::allocate(n, detail::find_allocator<allocator_type>(args...));
  }
  static void operator delete(void *p, std::size_t n) noexcept {
    frames::deallocate(p, n);
  }

  task get_return_object() noexcept {
    return task(std::coroutine_handle<promise_type>::from_promise(*this));
  }
  std::suspend_always initial_suspend() noexcept { return {}; }
  completion final_suspend() noexcept { return {}; }
  void unhandled_exception() noexcept {
    if constexpr (has_error<std::exception_ptr>)
      this->result.template emplace<std::exception_ptr>(
        std::current_exception());
    else
      std::terminate();
  }
  std::coroutine_handle<> unhandled_stopped() noexcept {
    return continuation_->stopped(*continuation_);
  }
  template <class E> completion yield_value(with_error<E> e) {
    static_assert(has_error<typename with_error<E>::type>,
      "with_error: the error type is not in the task's error_types");
    this->result.template emplace<typename with_error<E>::type>(
      std::move(e.error));
    return {};
  }

  template <class A> decltype(auto) await_transform(A &&a) {
    if constexpr (detail::is_change_scheduler<std::remove_cvref_t<A>>::value) {
      scheduler_type old =
        std::exchange(*sch_, scheduler_type(std::forward<A>(a).scheduler));
      return detail::await_sender<false>(
        then(exec::schedule(*sch_),
          [old = std::move(old)]() mutable noexcept { return std::move(old); }),
        *this);
    } else if constexpr (requires { std::forward<A>(a).as_awaitable(*this); }) {
      return std::forward<A>(a).as_awaitable(*this);
    } else {
      return detail::await_sender<affine>(std::forward<A>(a), *this);
    }
  }

  env_type get_env() const noexcept { return {this}; }
};

// The operation of a connected task. It owns the coroutine.
template <class T, class Environment>
template <class R>
class task<T, Environment>::state : detail::task_continuation {
  using rcvr_token = stop_token_of_t<env_of_t<R>>;
  // The receiver's stop token is used as is if it has the task's type.
  // Otherwise a stop request on it is forwarded to a source of the task's.
  static constexpr bool own_stop =
    !std::is_same_v<rcvr_token, stop_token_type> &&
    !unstoppable_token<rcvr_token>;
  struct forward {
    stop_source_type *source;
    void operator()() noexcept { source->request_stop(); }
  };
  struct stop_bridge {
    stop_source_type source;
    std::optional<stop_callback_for_t<rcvr_token, forward>> callback;
  };
  struct no_bridge {};

  std::coroutine_handle<promise_type> handle_;
  R rcvr_;
  [[no_unique_address]] std::conditional_t<own_stop, stop_bridge, no_bridge>
    stop_;

  void stop_forwarding() noexcept {
    if constexpr (own_stop)
      stop_.callback.reset();
  }

  static std::coroutine_handle<> complete(
    detail::task_continuation &c) noexcept {
    auto &self = static_cast<state &>(c);
    self.stop_forwarding();
    auto &result = self.handle_.promise().result;
    if (result.index() == 1) {
      if constexpr (std::is_void_v<T>)
        exec::set_value(std::move(self.rcvr_));
      else
        exec::set_value(
          std::move(self.rcvr_), std::move(std::get<1>(result).value));
    } else {
      std::visit(
        [&self](auto &e) noexcept {
          using E = std::decay_t<decltype(e)>;
          if constexpr (!std::is_same_v<E, std::monostate> &&
            !std::is_same_v<E, detail::task_value<T>>)
            exec::set_error(std::move(self.rcvr_), std::move(e));
        },
        result);
    }
    return std::noop_coroutine();
  }
  static std::coroutine_handle<> stopped(
    detail::task_continuation &c) noexcept {
    auto &self = static_cast<state &>(c);
    self.stop_forwarding();
    exec::set_stopped(std::move(self.rcvr_));
    return std::noop_coroutine();
  }

public:
  using operation_state_concept = operation_state_t;

  state(std::coroutine_handle<promise_type> h, R r)
      : detail::task_continuation{&complete, &stopped}, handle_(h),
        rcvr_(std::move(r)) {
    auto &p = handle_.promise();
    p.continuation_ = this;
    try {
      auto env = exec::get_env(rcvr_);
      if constexpr (requires { scheduler_type(get_scheduler(env)); })
        p.sch_.emplace(get_scheduler(env));
      else if constexpr (std::is_default_constructible_v<scheduler_type>)
        p.sch_.emplace();
      else if constexpr (std::is_constructible_v<scheduler_type,
                           inline_scheduler>)
        p.sch_.emplace(inline_scheduler{});
      else
        static_assert(std::is_void_v<R>,
          "task: the receiver's environment has no scheduler of "
          "scheduler_type");
    } catch (...) {
      handle_.destroy();
      throw;
    }
  }
  state(state &&) = delete;
  ~state() { handle_.destroy(); }

  void start() & noexcept {
    auto &p = handle_.promise();
    if constexpr (own_stop) {
      stop_.callback.emplace(
        get_stop_token(exec::get_env(rcvr_)), forward{&stop_.source});
      p.token_ = stop_.source.get_token();
    } else if constexpr (!unstoppable_token<rcvr_token>) {
      p.token_ = get_stop_token(exec::get_env(rcvr_));
    }
    handle_.resume();
  }

  // The scheduler the task runs on now, and completes on.
  const scheduler_type &current_scheduler() const noexcept {
    return *handle_.promise().sch_;
  }
};

} // namespace sg::exec
