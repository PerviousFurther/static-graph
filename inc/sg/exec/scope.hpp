#pragma once

// sg::exec: Async scopes.

#include "../core/graph.hpp"
#include "just.hpp"

#include <atomic>
#include <concepts>
#include <cstddef>
#include <exception>
#include <limits>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace sg::exec {

template <class T>
concept scope_token = std::copyable<T> && requires(const T &t) {
  { t.try_associate() } -> std::same_as<bool>;
  { t.disassociate() } noexcept;
  { t.wrap(just()) } -> sender;
};

namespace detail {
namespace desc {
// The steps before it see a stop token that fires when either token or the
// stop token below requests stop. Callbacks are registered at connect, as
// the layers before it have no start of their own, and removed on completion.
struct stop_when {
  inplace_stop_token token;
  static constexpr bool is_source = false;
  template <class In, class> using completions = In;
  template <class Env>
  using upstream_env = env<prop<get_stop_token_t, inplace_stop_token>, Env>;
  template <class T, class In> struct apply : layer<T, apply<T, In>> {
    using base = layer<T, apply<T, In>>;
    using outer_env = std::decay_t<env_of_t<const T &>>;
    using outer_token = stop_token_of_t<outer_env>;
    // With an unstoppable token below, token is used directly.
    static constexpr bool merge = !unstoppable_token<outer_token>;
    struct merged {
      inplace_stop_source source;
      std::optional<stop_callback_for_t<outer_token, forward_stop>> on_outer;
      std::optional<inplace_stop_callback<forward_stop>> on_token;
    };
    struct unmerged {};

    inplace_stop_token token;
    std::conditional_t<merge, merged, unmerged> stop;

    template <class... Rest>
    apply(stop_when d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...}, token(d.token) {
      if constexpr (merge) {
        stop.on_outer.emplace(
          get_stop_token(this->downstream_env()), forward_stop{&stop.source});
        stop.on_token.emplace(token, forward_stop{&stop.source});
      }
    }

    upstream_env<outer_env> get_env() const noexcept {
      inplace_stop_token t = token;
      if constexpr (merge)
        t = stop.source.get_token();
      return {prop<get_stop_token_t, inplace_stop_token>{{}, t},
        outer_env(this->downstream_env())};
    }
    template <class C, class... As> void complete(C, As &&...as) noexcept {
      if constexpr (merge) {
        stop.on_outer.reset();
        stop.on_token.reset();
      }
      this->pass(C{}, std::forward<As>(as)...);
    }
  };
};

// Holds sndr and, if the scope of token accepted it, an association; a copy
// tries to make its own. Started without one, it completes with set_stopped.
template <class Token, class S> struct associate {
  Token token;
  S sndr;
  bool associated;
  static constexpr bool is_source = true;
  template <class, class Env>
  using completions = concat_sigs_t<completion_signatures_of_t<S, Env>,
    completion_signatures<set_stopped_t()>>;

  associate(Token t, S s)
      : token(std::move(t)), sndr(std::move(s)),
        associated(token.try_associate()) {}
  associate(const associate &other)
      : token(other.token), sndr(other.sndr),
        associated(other.associated && token.try_associate()) {}
  associate(associate &&other) noexcept(std::is_nothrow_move_constructible_v<S>)
      : token(other.token), sndr(std::move(other.sndr)),
        associated(std::exchange(other.associated, false)) {}
  associate &operator=(const associate &) = delete;
  ~associate() {
    if (associated)
      token.disassociate();
  }

  // The association ends when the child completes, before passing it on.
  template <class T, class In> struct apply : layer<T, apply<T, In>> {
    using base = layer<T, apply<T, In>>;
    using receiver_type = forward_receiver<apply, env_of_t<const T &>>;
    struct child {
      connect_result_t<S, receiver_type> op;
      child(S &&s, receiver_type r) : op(exec::connect(std::move(s), r)) {}
    };
    Token token;
    bool associated = false;
    std::optional<child> c;

    template <class... Rest>
    apply(associate d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...}, token(d.token) {
      if (d.associated) {
        c.emplace(std::move(d.sndr), receiver_type{this});
        d.associated = false; // The association moves to this operation.
        associated = true;
      }
    }
    ~apply() {
      if (associated) {
        c.reset();
        token.disassociate();
      }
    }
    void start() noexcept {
      if (c)
        exec::start(c->op);
      else
        this->pass(set_stopped_t{});
    }
    template <class C, class... As> void complete(C, As &&...as) noexcept {
      associated = false;
      token.disassociate();
      this->pass(C{}, std::forward<As>(as)...);
    }
  };
};
} // namespace desc

template <class Token, class S>
using wrapped_t =
  std::remove_cvref_t<decltype(std::declval<const Token &>().wrap(
    std::declval<S>()))>;

// A spawned operation, freed when it completes.
template <class Token, class S, class Env> struct spawn_state {
  struct done_receiver {
    using receiver_concept = receiver_t;
    spawn_state *self;
    void set_value() && noexcept { self->done(); }
    void set_stopped() && noexcept { self->done(); }
    Env get_env() const noexcept { return self->environment; }
  };
  Token token;
  Env environment;
  connect_result_t<S, done_receiver> op;

  spawn_state(Token t, S &&s, Env e)
      : token(std::move(t)), environment(std::move(e)),
        op(exec::connect(std::move(s), done_receiver{this})) {}
  void done() noexcept {
    Token t = token;
    delete this;
    t.disassociate();
  }
};

// The shared state of spawn_future: the child stores its completion, and
// whichever of the child and the future finishes second takes the next step.
template <class Token, class S, class Env> struct future_state {
  using child_env = env<prop<get_stop_token_t, inplace_stop_token>, Env>;
  using sigs = completion_signatures_of_t<S, child_env>;
  using stored_sigs = concat_sigs_t<sig_map_t<sigs, decay_sig>,
    exception_sigs<nothrow_decay_v<sigs>>>;
  template <class... Ss>
  static auto variant_of(
    type_list<Ss...>) -> monostate_variant<typename sig<Ss>::tuple...>;
  enum : int { running, done, waiting, abandoned };

  struct child_receiver {
    using receiver_concept = receiver_t;
    future_state *self;
    template <class... Vs> void set_value(Vs &&...vs) && noexcept {
      self->complete(set_value_t{}, std::forward<Vs>(vs)...);
    }
    template <class E> void set_error(E &&e) && noexcept {
      self->complete(set_error_t{}, std::forward<E>(e));
    }
    void set_stopped() && noexcept { self->complete(set_stopped_t{}); }
    child_env get_env() const noexcept {
      return {
        prop<get_stop_token_t, inplace_stop_token>{{}, self->stop.get_token()},
        self->environment};
    }
  };

  inplace_stop_source stop;
  Token token;
  Env environment;
  decltype(variant_of(stored_sigs{})) result;
  std::atomic<int> state{running};
  void (*notify)(void *) noexcept = nullptr;
  void *consumer = nullptr;
  connect_result_t<S, child_receiver> op;

  future_state(Token t, S &&s, Env e)
      : token(std::move(t)), environment(std::move(e)),
        op(exec::connect(std::move(s), child_receiver{this})) {}

  template <class C, class... As> void complete(C, As &&...as) noexcept {
    try_invoke<nothrow_decay_v<sigs>>(
      [&] {
        result.template emplace<std::tuple<C, std::decay_t<As>...>>(
          C{}, std::forward<As>(as)...);
      },
      [this](auto e) noexcept {
        result.template emplace<std::tuple<set_error_t, std::exception_ptr>>(
          set_error_t{}, std::move(e));
      });
    token.disassociate();
    switch (state.exchange(done, std::memory_order_acq_rel)) {
    case waiting:
      notify(consumer);
      break;
    case abandoned:
      delete this;
      break;
    }
  }
  // The future was started; fn(c) runs once the result is ready, maybe now.
  void wait(void (*fn)(void *) noexcept, void *c) noexcept {
    notify = fn;
    consumer = c;
    int expected = running;
    if (!state.compare_exchange_strong(
          expected, waiting, std::memory_order_acq_rel))
      fn(c);
  }
  // The future was dropped unstarted: stop the child and let go.
  void abandon() noexcept {
    stop.request_stop();
    int expected = running;
    if (!state.compare_exchange_strong(
          expected, abandoned, std::memory_order_acq_rel))
      delete this;
  }
};

namespace desc {
// The future sender: owns the state, which is null if spawning was refused.
template <class State> struct future {
  State *state;
  static constexpr bool is_source = true;
  template <class, class>
  using completions = concat_sigs_t<typename State::stored_sigs,
    completion_signatures<set_stopped_t()>>;

  explicit future(State *s) noexcept : state(s) {}
  future(future &&other) noexcept
      : state(std::exchange(other.state, nullptr)) {}
  ~future() {
    if (state)
      state->abandon();
  }

  template <class T, class In> struct apply : layer<T, apply<T, In>> {
    using base = layer<T, apply<T, In>>;
    using outer_env = std::decay_t<env_of_t<const T &>>;
    State *state;
    bool started = false;
    std::optional<stop_callback_for_t<stop_token_of_t<outer_env>, forward_stop>>
      on_stop;

    template <class... Rest>
    apply(future d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...},
          state(std::exchange(d.state, nullptr)) {}
    ~apply() {
      if (!state)
        return;
      if (started)
        delete state;
      else
        state->abandon();
    }
    void start() noexcept {
      if (!state) {
        this->pass(set_stopped_t{});
        return;
      }
      started = true;
      on_stop.emplace(
        get_stop_token(this->downstream_env()), forward_stop{&state->stop});
      state->wait(
        [](void *self) noexcept { static_cast<apply *>(self)->deliver(); },
        this);
    }

  private:
    void deliver() noexcept {
      on_stop.reset();
      std::visit(
        [this](auto &r) noexcept {
          if constexpr (!std::is_same_v<std::decay_t<decltype(r)>,
                          std::monostate>)
            std::apply(
              [this](auto tag, auto &...xs) noexcept {
                this->pass(tag, std::move(xs)...);
              },
              r);
        },
        state->result);
    }
  };
};
} // namespace desc

struct join_waiter {
  join_waiter *next = nullptr;
  void (*resume)(join_waiter *) noexcept = nullptr;
};

// The association count of a counting scope and its close and join flags, in
// one atomic word. Join waiters form a lock-free stack, drained once the
// count reaches zero after a join started.
class scope_count {
  static constexpr std::size_t closed = 1, joining = 2, joined = 4, used = 8,
                               unit = 16;
  std::atomic<std::size_t> state_{0};
  std::atomic<join_waiter *> waiters_{nullptr};

  void drain() noexcept {
    auto *w = waiters_.exchange(nullptr, std::memory_order_acq_rel);
    while (w) {
      auto *next = w->next; // resume may destroy w
      w->resume(w);
      w = next;
    }
  }

public:
  static constexpr std::size_t max_associations =
    std::numeric_limits<std::size_t>::max() / unit;

  scope_count() = default;
  scope_count(const scope_count &) = delete;
  scope_count &operator=(const scope_count &) = delete;
  // A scope that was used must be joined before it is destroyed.
  ~scope_count() {
    auto s = state_.load(std::memory_order_relaxed);
    if ((s & used) && !(s & joined))
      std::terminate();
  }

  bool try_associate() noexcept {
    auto s = state_.load(std::memory_order_relaxed);
    do {
      if ((s & (closed | joined)) || s / unit == max_associations)
        return false;
    } while (!state_.compare_exchange_weak(
      s, (s + unit) | used, std::memory_order_relaxed));
    return true;
  }
  void disassociate() noexcept {
    auto s = state_.load(std::memory_order_relaxed);
    std::size_t n;
    do {
      n = s - unit;
      if (n / unit == 0 && (n & joining))
        n |= joined;
    } while (!state_.compare_exchange_weak(
      s, n, std::memory_order_acq_rel, std::memory_order_relaxed));
    if (n & joined)
      drain();
  }
  void close() noexcept { state_.fetch_or(closed, std::memory_order_relaxed); }
  // Returns true if the scope is already joined. Otherwise w is resumed once
  // it is, which may happen before this returns.
  bool join(join_waiter &w) noexcept {
    auto s = state_.load(std::memory_order_acquire);
    if (s & joined)
      return true;
    w.next = waiters_.load(std::memory_order_relaxed);
    while (!waiters_.compare_exchange_weak(
      w.next, &w, std::memory_order_release, std::memory_order_relaxed)) {
    }
    std::size_t n;
    do {
      n = s | joining;
      if (n / unit == 0)
        n |= joined;
    } while (!state_.compare_exchange_weak(
      s, n, std::memory_order_acq_rel, std::memory_order_relaxed));
    if (n & joined)
      drain();
    return false;
  }
};

namespace desc {
// Completes with set_value once the scope's count reaches zero, on the thread
// that ends the last association (or inline).
struct join {
  scope_count *scope;
  static constexpr bool is_source = true;
  template <class, class>
  using completions = completion_signatures<set_value_t()>;
  template <class T, class In>
  struct apply : layer<T, apply<T, In>>, join_waiter {
    using base = layer<T, apply<T, In>>;
    scope_count *scope;
    template <class... Rest>
    apply(join d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...}, scope(d.scope) {}
    void start() noexcept {
      resume = [](join_waiter *w) noexcept {
        static_cast<apply *>(w)->pass(set_value_t{});
      };
      if (scope->join(*this))
        this->pass(set_value_t{});
    }
  };
};
} // namespace desc
} // namespace detail

// Runs s inside the scope of token, or completes with set_stopped if the
// scope refuses new work.
struct associate_t : detail::adaptor<associate_t> {
  using adaptor::operator();
  template <sender S, scope_token Token>
  auto operator()(S &&s, Token token) const {
    using wrapped = detail::wrapped_t<Token, S>;
    return basic_sender{detail::desc::associate<Token, wrapped>{
      token, token.wrap(std::forward<S>(s))}};
  }
};
inline constexpr associate_t associate{};

// Starts s inside the scope of token and forgets it; s must complete with
// set_value() or set_stopped(). Does nothing if the scope refuses new work.
struct spawn_t {
  template <sender S, scope_token Token, class Env = env<>>
  void operator()(S &&s, Token token, Env e = {}) const {
    using wrapped = detail::wrapped_t<Token, S>;
    using allowed = completion_signatures<set_value_t(), set_stopped_t()>;
    static_assert(std::is_same_v<detail::concat_sigs_t<allowed,
                                   completion_signatures_of_t<wrapped, Env>>,
                    allowed>,
      "spawn requires a sender that completes with set_value() or "
      "set_stopped()");
    wrapped w = token.wrap(std::forward<S>(s));
    if (!token.try_associate())
      return;
    detail::spawn_state<Token, wrapped, Env> *state;
    try {
      state = new detail::spawn_state<Token, wrapped, Env>(
        token, std::move(w), std::move(e));
    } catch (...) {
      token.disassociate();
      throw;
    }
    exec::start(state->op);
  }
};
inline constexpr spawn_t spawn{};

// Starts s inside the scope of token now and returns a sender of its result.
// Dropping the sender unstarted requests stop on s. If the scope refuses new
// work, the sender completes with set_stopped.
struct spawn_future_t {
  template <sender S, scope_token Token, class Env = env<>>
  auto operator()(S &&s, Token token, Env e = {}) const {
    using wrapped = detail::wrapped_t<Token, S>;
    using state_type = detail::future_state<Token, wrapped, Env>;
    wrapped w = token.wrap(std::forward<S>(s));
    state_type *state = nullptr;
    if (token.try_associate()) {
      try {
        state = new state_type(token, std::move(w), std::move(e));
      } catch (...) {
        token.disassociate();
        throw;
      }
      exec::start(state->op);
    }
    return basic_sender{detail::desc::future<state_type>{state}};
  }
};
inline constexpr spawn_future_t spawn_future{};

// Counts the work associated with it. close() refuses new work; join() is a
// sender that completes once the count reaches zero, after which the scope
// refuses new work too. A scope that was used must be joined before it is
// destroyed.
class simple_counting_scope {
  detail::scope_count count_;

public:
  static constexpr std::size_t max_associations =
    detail::scope_count::max_associations;
  struct token {
    simple_counting_scope *scope;
    template <sender S> S &&wrap(S &&s) const noexcept {
      return std::forward<S>(s);
    }
    bool try_associate() const noexcept {
      return scope->count_.try_associate();
    }
    void disassociate() const noexcept { scope->count_.disassociate(); }
  };

  token get_token() noexcept { return {this}; }
  void close() noexcept { count_.close(); }
  auto join() noexcept { return basic_sender{detail::desc::join{&count_}}; }
};

// A simple_counting_scope that can also request stop on its work.
class counting_scope {
  detail::scope_count count_;
  inplace_stop_source stop_;

public:
  static constexpr std::size_t max_associations =
    detail::scope_count::max_associations;
  struct token {
    counting_scope *scope;
    template <sender S> auto wrap(S &&s) const {
      return detail::append(
        std::forward<S>(s), detail::desc::stop_when{scope->stop_.get_token()});
    }
    bool try_associate() const noexcept {
      return scope->count_.try_associate();
    }
    void disassociate() const noexcept { scope->count_.disassociate(); }
  };

  token get_token() noexcept { return {this}; }
  void close() noexcept { count_.close(); }
  void request_stop() noexcept { stop_.request_stop(); }
  auto join() noexcept { return basic_sender{detail::desc::join{&count_}}; }
};

} // namespace sg::exec
