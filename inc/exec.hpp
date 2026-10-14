#pragma once

// Sender/receiver execution built on sg::graph, following C++26
// std::execution names and semantics.
//
// A sender is a list of descriptors. connect builds one sg::graph in
// receiver-first order, so every adaptor layer derives from its downstream
// layer T. Layers pass completions down with T::complete(Tag, args...);
// set_value, set_error and set_stopped are only used at the boundaries, to
// call the connected receiver and to receive from nested operations. A layer
// reads its environment with downstream_env(), which finds the nearest get_env
// below it: the receiver's, or that of a step changing it (write_env is a
// graph state holding its env). Completion signatures are computed per
// descriptor, in the environment that step sees, and handed to each layer as
// In, so intermediate storage is typed. A sender whose value only exists at
// run time (the result of let_*'s function) or that runs beside the chain
// (when_all's children, schedule senders) is connected as a separate operation
// held by its layer, with a receiver that calls back into that layer.
//
// Coroutines and senders meet in two places. An awaitable is a sender: connect
// runs a small coroutine that awaits it. A sender is awaitable in a coroutine
// whose promise uses as_awaitable (with_awaitable_senders, task): its
// operation is connected into the awaiter, in the coroutine frame.
//
// Not provided: tag_invoke customization, domains and transform_sender,
// sequence senders.
//
// Keep an operation alive until it completes. Receivers must be noexcept and
// must consume their arguments before destroying the operation that sent them.

#include "graph.hpp"

#include <algorithm>
#include <atomic>
#include <concepts>
#include <condition_variable>
#include <coroutine>
#include <cstddef>
#include <exception>
#include <execution>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace sg::exec {

// ---- Tags and core customization points ---------------------------------

struct sender_t {};
struct receiver_t {};
struct operation_state_t {};
struct scheduler_t {};

// The completion tags name the channels in completion signatures. Calling one
// calls the receiver member of the same name.
struct set_value_t {
  template <class R, class... Vs>
  constexpr void operator()(R &&r, Vs &&...vs) const noexcept {
    static_assert(
      noexcept(std::forward<R>(r).set_value(std::forward<Vs>(vs)...)),
      "set_value must be noexcept");
    std::forward<R>(r).set_value(std::forward<Vs>(vs)...);
  }
};
struct set_error_t {
  template <class R, class E>
  constexpr void operator()(R &&r, E &&e) const noexcept {
    static_assert(noexcept(std::forward<R>(r).set_error(std::forward<E>(e))),
      "set_error must be noexcept");
    std::forward<R>(r).set_error(std::forward<E>(e));
  }
};
struct set_stopped_t {
  template <class R> constexpr void operator()(R &&r) const noexcept {
    static_assert(noexcept(std::forward<R>(r).set_stopped()),
      "set_stopped must be noexcept");
    std::forward<R>(r).set_stopped();
  }
};
inline constexpr set_value_t set_value{};
inline constexpr set_error_t set_error{};
inline constexpr set_stopped_t set_stopped{};

struct start_t {
  template <class O> constexpr void operator()(O &op) const noexcept {
    static_assert(noexcept(op.start()), "start must be noexcept");
    op.start();
  }
};
inline constexpr start_t start{};

struct schedule_t {
  template <class Sch>
  constexpr auto operator()(Sch &&sch) const
    noexcept(noexcept(std::forward<Sch>(sch).schedule()))
      -> decltype(std::forward<Sch>(sch).schedule()) {
    return std::forward<Sch>(sch).schedule();
  }
};
inline constexpr schedule_t schedule{};

// ---- Stop tokens --------------------------------------------------------

class never_stop_token {
  struct callback {
    template <class C> explicit callback(never_stop_token, C &&) noexcept {}
  };

public:
  template <class> using callback_type = callback;
  static constexpr bool stop_requested() noexcept { return false; }
  static constexpr bool stop_possible() noexcept { return false; }
  friend constexpr bool operator==(
    never_stop_token, never_stop_token) noexcept = default;
};

template <class Token, class CB> struct stop_callback_for {
  using type = typename Token::template callback_type<CB>;
};
template <class CB> struct stop_callback_for<std::stop_token, CB> {
  using type = std::stop_callback<CB>;
};
template <class Token, class CB>
using stop_callback_for_t = typename stop_callback_for<Token, CB>::type;

class inplace_stop_source;
template <class CB> class inplace_stop_callback;

namespace detail {
struct stop_callback_node {
  void (*execute)(stop_callback_node *) noexcept;
  stop_callback_node *next = nullptr;
  stop_callback_node **prev = nullptr; // null when not linked
  bool *removed = nullptr; // set while the callback runs
  std::atomic<bool> done{false};
};
} // namespace detail

class inplace_stop_token {
  const inplace_stop_source *source_ = nullptr;
  friend inplace_stop_source;
  template <class> friend class inplace_stop_callback;
  explicit inplace_stop_token(const inplace_stop_source *s) noexcept
      : source_(s) {}

public:
  template <class CB> using callback_type = inplace_stop_callback<CB>;
  inplace_stop_token() = default;
  bool stop_requested() const noexcept;
  bool stop_possible() const noexcept { return source_ != nullptr; }
  void swap(inplace_stop_token &other) noexcept {
    std::swap(source_, other.source_);
  }
  friend bool operator==(
    const inplace_stop_token &, const inplace_stop_token &) = default;
};

// Callbacks run on the thread that requests stop. A callback destroyed on
// another thread while it runs waits for it; one destroyed by itself does not.
// One atomic word holds the stop bit and a spin bit guarding the list.
class inplace_stop_source {
  static constexpr unsigned stop_bit = 1, lock_bit = 2;
  mutable std::atomic<unsigned> state_{0};
  mutable detail::stop_callback_node *head_ = nullptr;
  std::thread::id requester_;
  template <class> friend class inplace_stop_callback;

  // Returns the state as it was before locking.
  unsigned lock() const noexcept {
    unsigned s = state_.load(std::memory_order_relaxed);
    for (;;) {
      if (s & lock_bit) {
        std::this_thread::yield();
        s = state_.load(std::memory_order_relaxed);
      } else if (state_.compare_exchange_weak(s, s | lock_bit,
                   std::memory_order_acquire, std::memory_order_relaxed)) {
        return s;
      }
    }
  }
  void unlock(unsigned s) const noexcept {
    state_.store(s, std::memory_order_release);
  }

  bool add(detail::stop_callback_node *cb) const noexcept {
    unsigned s = lock();
    if (!(s & stop_bit)) {
      cb->next = head_;
      cb->prev = &head_;
      if (head_)
        head_->prev = &cb->next;
      head_ = cb;
    }
    unlock(s);
    return !(s & stop_bit);
  }

  void remove(detail::stop_callback_node *cb) const noexcept {
    unsigned s = lock();
    bool linked = cb->prev != nullptr;
    if (linked) {
      *cb->prev = cb->next;
      if (cb->next)
        cb->next->prev = cb->prev;
    }
    unlock(s);
    if (linked || cb->done.load(std::memory_order_acquire))
      return;
    // request_stop unlinked cb and is running it.
    if (requester_ == std::this_thread::get_id())
      *cb->removed = true;
    else
      while (!cb->done.load(std::memory_order_acquire))
        std::this_thread::yield();
  }

public:
  inplace_stop_source() = default;
  inplace_stop_source(const inplace_stop_source &) = delete;
  inplace_stop_source &operator=(const inplace_stop_source &) = delete;

  inplace_stop_token get_token() const noexcept {
    return inplace_stop_token{this};
  }
  static constexpr bool stop_possible() noexcept { return true; }
  bool stop_requested() const noexcept {
    return state_.load(std::memory_order_acquire) & stop_bit;
  }

  bool request_stop() noexcept {
    if (lock() & stop_bit) {
      unlock(stop_bit);
      return false;
    }
    requester_ = std::this_thread::get_id();
    while (auto *cb = head_) {
      head_ = cb->next;
      if (head_)
        head_->prev = &head_;
      cb->prev = nullptr;
      bool removed = false;
      cb->removed = &removed;
      unlock(stop_bit);
      cb->execute(cb);
      if (!removed)
        cb->done.store(true, std::memory_order_release);
      lock();
    }
    unlock(stop_bit);
    return true;
  }
};

inline bool inplace_stop_token::stop_requested() const noexcept {
  return source_ && source_->stop_requested();
}

template <class CB> class inplace_stop_callback : detail::stop_callback_node {
  CB callback_;
  const inplace_stop_source *source_;

  static void run(detail::stop_callback_node *node) noexcept {
    std::move(static_cast<inplace_stop_callback *>(node)->callback_)();
  }

public:
  template <class Init>
  explicit inplace_stop_callback(inplace_stop_token token,
    Init &&init) noexcept(std::is_nothrow_constructible_v<CB, Init>)
      : stop_callback_node{&run}, callback_(std::forward<Init>(init)),
        source_(token.source_) {
    if (source_ && !source_->add(this)) {
      source_ = nullptr;
      std::move(callback_)();
    }
  }
  inplace_stop_callback(const inplace_stop_callback &) = delete;
  inplace_stop_callback &operator=(const inplace_stop_callback &) = delete;
  ~inplace_stop_callback() {
    if (source_)
      source_->remove(this);
  }
};
template <class CB>
inplace_stop_callback(inplace_stop_token, CB) -> inplace_stop_callback<CB>;

namespace detail {
// A stop callback that requests stop on another source.
struct forward_stop {
  inplace_stop_source *source;
  void operator()() noexcept { source->request_stop(); }
};
} // namespace detail

// ---- Environments and queries -------------------------------------------

template <class Q, class V> struct prop {
  [[no_unique_address]] Q query_;
  V value_;
  constexpr const V &query(Q) const noexcept { return value_; }
};
template <class Q, class V> prop(Q, V) -> prop<Q, V>;

namespace detail {
template <class E, class Q>
concept has_query = requires(const E &e, Q q) { e.query(q); };
} // namespace detail

// Queries are answered by the first environment that supports them.
template <class... Es> struct env {
  std::tuple<Es...> envs;
  constexpr env(Es... es) : envs(std::move(es)...) {}

private:
  template <class Q> static constexpr std::size_t index_of() noexcept {
    constexpr bool found[] = {detail::has_query<Es, Q>..., true};
    std::size_t i = 0;
    while (!found[i])
      ++i;
    return i;
  }

public:
  template <class Q>
    requires(index_of<Q>() < sizeof...(Es))
  constexpr decltype(auto) query(Q q) const noexcept {
    return std::get<index_of<Q>()>(envs).query(q);
  }
};
template <class... Es> env(Es...) -> env<Es...>;

struct get_env_t {
  template <class T>
  constexpr decltype(auto) operator()(const T &t) const noexcept {
    if constexpr (requires { t.get_env(); })
      return t.get_env();
    else
      return env<>{};
  }
};
inline constexpr get_env_t get_env{};
template <class T> using env_of_t = decltype(get_env(std::declval<T>()));

struct get_stop_token_t {
  template <class E> constexpr auto operator()(const E &e) const noexcept {
    if constexpr (detail::has_query<E, get_stop_token_t>)
      return e.query(*this);
    else
      return never_stop_token{};
  }
};
inline constexpr get_stop_token_t get_stop_token{};
template <class T>
using stop_token_of_t =
  std::decay_t<decltype(get_stop_token(std::declval<const T &>()))>;

#define SG_EXEC_QUERY(name)                                                    \
  struct name##_t {                                                            \
    template <class E>                                                         \
    constexpr auto operator()(const E &e) const noexcept                       \
      -> decltype(e.query(std::declval<const name##_t &>())) {                 \
      return e.query(*this);                                                   \
    }                                                                          \
  };                                                                           \
  inline constexpr name##_t name {}
SG_EXEC_QUERY(get_scheduler);
SG_EXEC_QUERY(get_delegation_scheduler);
SG_EXEC_QUERY(get_allocator);
#undef SG_EXEC_QUERY

template <class Tag> struct get_completion_scheduler_t {
  template <class E>
  constexpr auto operator()(const E &e) const noexcept
    -> decltype(e.query(std::declval<const get_completion_scheduler_t &>())) {
    return e.query(*this);
  }
};
template <class Tag>
inline constexpr get_completion_scheduler_t<Tag> get_completion_scheduler{};

enum class forward_progress_guarantee { concurrent, parallel, weakly_parallel };

struct get_forward_progress_guarantee_t {
  template <class Sch>
  constexpr forward_progress_guarantee operator()(
    const Sch &sch) const noexcept {
    if constexpr (detail::has_query<Sch, get_forward_progress_guarantee_t>)
      return sch.query(*this);
    else
      return forward_progress_guarantee::weakly_parallel;
  }
};
inline constexpr get_forward_progress_guarantee_t
  get_forward_progress_guarantee{};

// The most tasks a scheduler's resource runs at once: a fixed property, not
// how many are idle. Not a standard query; parallel bulk splits its work by
// it.
struct get_max_concurrency_t {
  template <class Sch>
    requires detail::has_query<Sch, get_max_concurrency_t>
  constexpr std::size_t operator()(const Sch &sch) const noexcept {
    return sch.query(*this);
  }
};
inline constexpr get_max_concurrency_t get_max_concurrency{};

namespace detail {
// Attributes of a sender whose value and stopped completions run on sch.
template <class Sch> struct sched_attrs {
  Sch sch;
  template <class Tag>
    requires(
      std::is_same_v<Tag, set_value_t> || std::is_same_v<Tag, set_stopped_t>)
  constexpr Sch query(get_completion_scheduler_t<Tag>) const noexcept {
    return sch;
  }
};
} // namespace detail

// ---- Awaitables and connect ---------------------------------------------

namespace detail {
// The operand of co_await after applying operator co_await, if any.
template <class A> decltype(auto) co_await_operand(A &&a) {
  if constexpr (requires { std::forward<A>(a).operator co_await(); })
    return std::forward<A>(a).operator co_await();
  else if constexpr (requires { operator co_await(std::forward<A>(a)); })
    return operator co_await(std::forward<A>(a));
  else
    return std::forward<A>(a);
}
// The awaiter of co_await a in a coroutine whose promise is P.
template <class A, class P> decltype(auto) get_awaiter(A &&a, P *p) {
  if constexpr (requires { p->await_transform(std::forward<A>(a)); })
    return co_await_operand(p->await_transform(std::forward<A>(a)));
  else
    return co_await_operand(std::forward<A>(a));
}

template <class W, class P>
concept awaiter = requires(W &w, std::coroutine_handle<P> h) {
  w.await_ready() ? 0 : 0;
  w.await_suspend(h);
  w.await_resume();
};
template <class A, class P>
concept is_awaitable = requires(A &&a, P *p) {
  { get_awaiter(std::forward<A>(a), p) } -> awaiter<P>;
};
template <class A, class P>
using await_result_t =
  decltype(std::declval<decltype(get_awaiter(
             std::declval<A>(), static_cast<P *>(nullptr)))>()
             .await_resume());

// A promise without await_transform, for checking plain awaitables.
struct plain_promise {};

// Awaits what defines as_awaitable(promise) through it, the rest as is.
template <class Promise> struct with_await_transform {
  template <class A> decltype(auto) await_transform(A &&a) {
    if constexpr (requires {
                    std::forward<A>(a).as_awaitable(
                      static_cast<Promise &>(*this));
                  })
      return std::forward<A>(a).as_awaitable(static_cast<Promise &>(*this));
    else
      return std::forward<A>(a);
  }
};

// The promise an awaitable is checked against when it is used as a sender in
// environment Env. Only used unevaluated.
template <class Env>
struct env_promise : with_await_transform<env_promise<Env>> {
  const Env &get_env() const noexcept;
  std::coroutine_handle<> unhandled_stopped() noexcept;
};

template <class R> class awaitable_operation;
// Connects an awaitable to r through a coroutine that awaits it and completes
// r with the result. Defined with the coroutine support at the end.
template <class A, class R> awaitable_operation<R> connect_awaitable(A a, R r);
} // namespace detail

struct connect_t {
  template <class S, class R>
  constexpr auto operator()(S &&s, R &&r) const
    noexcept(noexcept(std::forward<S>(s).connect(std::forward<R>(
      r)))) -> decltype(std::forward<S>(s).connect(std::forward<R>(r))) {
    return std::forward<S>(s).connect(std::forward<R>(r));
  }
  // An awaitable without connect: the operation is a coroutine awaiting it.
  template <class S, class R>
    requires(!requires(S &&s, R &&r) {
      std::forward<S>(s).connect(std::forward<R>(r));
    }) &&
    detail::is_awaitable<S, detail::env_promise<std::decay_t<env_of_t<R>>>>
  auto operator()(S &&s, R &&r) const {
    return detail::connect_awaitable(std::remove_cvref_t<S>(std::forward<S>(s)),
      std::remove_cvref_t<R>(std::forward<R>(r)));
  }
};
inline constexpr connect_t connect{};

// ---- Completion signatures ----------------------------------------------

// A set of Tag(Args...) signatures. Being a type_list, it works with the
// type_list operations in graph.hpp.
template <class... Sigs> using completion_signatures = type_list<Sigs...>;

namespace detail {
// Union of signature sets, in first-seen order.
template <class... Sets>
using concat_sigs_t =
  typename type_list_unique<typename type_list_cat<Sets...>::type>::type;

template <class Set, template <class> class F> struct sig_map;
template <class... Ss, template <class> class F>
struct sig_map<type_list<Ss...>, F> {
  using type = concat_sigs_t<typename F<Ss>::type...>;
};
// Replaces every signature S in Set with the set F<S>::type. Bind extra
// parameters of F with type_bind_back.
template <class Set, template <class> class F>
using sig_map_t = typename sig_map<Set, F>::type;

template <class Sig> struct sig;
template <class Tag, class... As> struct sig<Tag(As...)> {
  using tag = Tag;
  using decayed = Tag(std::decay_t<As>...);
  using tuple = std::tuple<Tag, std::decay_t<As>...>;
  static constexpr bool nothrow_decay =
    (std::is_nothrow_constructible_v<std::decay_t<As>, As> && ...);
};
template <class S, class Tag>
inline constexpr bool sig_is = std::is_same_v<typename sig<S>::tag, Tag>;

// set_value_t(R), or set_value_t() for void.
template <class R> struct value_sig {
  using type = set_value_t(R);
};
template <> struct value_sig<void> {
  using type = set_value_t();
};

template <class S> struct decay_sig {
  using type = type_list<typename sig<S>::decayed>;
};
template <class S> struct non_value_sig {
  using type =
    std::conditional_t<sig_is<S, set_value_t>, type_list<>, type_list<S>>;
};
template <class S> struct error_sig {
  using type = std::conditional_t<sig_is<S, set_error_t>,
    typename decay_sig<S>::type, type_list<>>;
};

template <class... Ss> constexpr bool nothrow_decay(type_list<Ss...>) {
  return (sig<Ss>::nothrow_decay && ...);
}
template <class Set>
inline constexpr bool nothrow_decay_v = nothrow_decay(Set{});

// set_error_t(exception_ptr), unless the step cannot throw.
template <bool Nothrow>
using exception_sigs = std::conditional_t<Nothrow, type_list<>,
  type_list<set_error_t(std::exception_ptr)>>;

// Variant<Tuple<As...>...> over the Tag(As...) signatures in Set.
template <class Tag, template <class...> class Tuple> struct select_sigs {
  template <class S> struct one {
    using type = type_list<>;
  };
  template <class... As> struct one<Tag(As...)> {
    using type = type_list<Tuple<As...>>;
  };
};
template <class Tag, class Set, template <class...> class Tuple,
  template <class...> class Variant>
using gather_signatures_t = typename type_list_apply<Variant,
  sig_map_t<Set, select_sigs<Tag, Tuple>::template one>>::type;

template <class... Ts> struct single_type;
template <class T> struct single_type<T> {
  using type = T;
};
template <class... Ts> using single_t = typename single_type<Ts...>::type;

template <class... Ts>
using monostate_variant = std::variant<std::monostate, Ts...>;

struct empty_variant {};
template <class... Ts> struct variant_or_empty_impl {
  using type = std::variant<Ts...>;
};
template <> struct variant_or_empty_impl<> {
  using type = empty_variant;
};
} // namespace detail

template <class... Ts> using decayed_tuple = std::tuple<std::decay_t<Ts>...>;
template <class... Ts>
using variant_or_empty = typename detail::variant_or_empty_impl<Ts...>::type;

struct get_completion_signatures_t {
  template <class S, class Env = env<>>
  constexpr auto operator()(S &&s, Env &&e = {}) const noexcept {
    using promise = detail::env_promise<std::decay_t<Env>>;
    if constexpr (requires {
                    typename std::remove_cvref_t<S>::completion_signatures;
                  })
      return typename std::remove_cvref_t<S>::completion_signatures{};
    else if constexpr (!requires {
                         std::forward<S>(s).get_completion_signatures(
                           std::forward<Env>(e));
                       } && detail::is_awaitable<S, promise>)
      return completion_signatures<
        typename detail::value_sig<detail::await_result_t<S, promise>>::type,
        set_error_t(std::exception_ptr), set_stopped_t()>{};
    else
      return std::forward<S>(s).get_completion_signatures(std::forward<Env>(e));
  }
};
inline constexpr get_completion_signatures_t get_completion_signatures{};

template <class S, class Env = env<>>
using completion_signatures_of_t =
  decltype(get_completion_signatures(std::declval<S>(), std::declval<Env>()));

template <class S, class Env = env<>,
  template <class...> class Tuple = decayed_tuple,
  template <class...> class Variant = variant_or_empty>
using value_types_of_t = detail::gather_signatures_t<set_value_t,
  completion_signatures_of_t<S, Env>, Tuple, Variant>;
template <class S, class Env = env<>,
  template <class...> class Variant = variant_or_empty>
using error_types_of_t = detail::gather_signatures_t<set_error_t,
  completion_signatures_of_t<S, Env>, detail::single_t, Variant>;
template <class S, class Env = env<>>
inline constexpr bool sends_stopped =
  type_list_size<detail::gather_signatures_t<set_stopped_t,
    completion_signatures_of_t<S, Env>, type_list, type_list>>::value != 0;

// ---- Concepts -----------------------------------------------------------

namespace detail {
template <class S>
concept has_sender_concept =
  std::derived_from<typename std::remove_cvref_t<S>::sender_concept, sender_t>;
} // namespace detail

// Awaitables are senders too: connect awaits them in a coroutine.
template <class S>
concept sender = (detail::has_sender_concept<S> ||
                   detail::is_awaitable<S, detail::env_promise<env<>>>) &&
  std::move_constructible<std::remove_cvref_t<S>>;

template <class S, class Env = env<>>
concept sender_in =
  sender<S> && requires { typename completion_signatures_of_t<S, Env>; };

template <class R>
concept receiver =
  std::derived_from<typename std::remove_cvref_t<R>::receiver_concept,
    receiver_t> &&
  std::move_constructible<std::remove_cvref_t<R>>;

template <class O>
concept operation_state =
  std::destructible<O> && std::is_object_v<O> && requires(O &o) {
    { o.start() } noexcept;
  };

template <class Sch>
concept scheduler =
  requires(Sch &&sch) {
    { schedule(std::forward<Sch>(sch)) } -> sender;
  } && std::equality_comparable<std::remove_cvref_t<Sch>> &&
  std::copy_constructible<std::remove_cvref_t<Sch>>;

template <class S, class R>
concept sender_to = sender_in<S, env_of_t<R>> && receiver<R> &&
  requires(S &&s, R &&r) { connect(std::forward<S>(s), std::forward<R>(r)); };

namespace detail {
template <class R, class Sig> inline constexpr bool accepts_v = false;
template <class R, class... As>
inline constexpr bool accepts_v<R, set_value_t(As...)> =
  requires(R &&r, As &&...as) {
    { std::forward<R>(r).set_value(std::forward<As>(as)...) } noexcept;
  };
template <class R, class E>
inline constexpr bool accepts_v<R, set_error_t(E)> = requires(R &&r, E &&e) {
  { std::forward<R>(r).set_error(std::forward<E>(e)) } noexcept;
};
template <class R>
inline constexpr bool accepts_v<R, set_stopped_t()> = requires(R &&r) {
  { std::forward<R>(r).set_stopped() } noexcept;
};
template <class R, class Sigs> inline constexpr bool accepts_all_v = false;
template <class R, class... Ss>
inline constexpr bool accepts_all_v<R, type_list<Ss...>> =
  (accepts_v<R, Ss> && ...);
} // namespace detail

template <class R, class Sigs>
concept receiver_of =
  receiver<R> && detail::accepts_all_v<std::remove_cvref_t<R>, Sigs>;

template <class T>
concept stoppable_token =
  std::copyable<T> && std::equality_comparable<T> && requires(const T &t) {
    { t.stop_requested() } noexcept -> std::same_as<bool>;
    { t.stop_possible() } noexcept -> std::same_as<bool>;
  };
template <class T>
concept unstoppable_token =
  stoppable_token<T> && requires { requires(!T::stop_possible()); };

template <class S, class R>
using connect_result_t =
  decltype(connect(std::declval<S>(), std::declval<R>()));
template <class Sch>
using schedule_result_t = decltype(schedule(std::declval<Sch>()));

// ---- Graph layers -------------------------------------------------------

namespace detail {

// Calls fn. If fn may throw and does, calls on_error with the exception and
// returns false.
template <bool Nothrow, class Fn, class OnError>
bool try_invoke(Fn &&fn, OnError &&on_error) noexcept {
  if constexpr (Nothrow) {
    fn();
    return true;
  } else {
    try {
      fn();
      return true;
    } catch (...) {
      on_error(std::current_exception());
      return false;
    }
  }
}

// Base of every adaptor layer. T is the downstream layer. By default a layer
// passes every completion down unchanged.
template <class T, class Self> struct layer : sg::node::apply<T, Self> {
  using base = sg::node::apply<T, Self>;
  template <class... Rest>
  constexpr layer(Rest &&...rest)
      : base{sg::node{}, std::forward<Rest>(rest)...} {}

  template <class Tag, class... As> void complete(Tag, As &&...as) noexcept {
    pass(Tag{}, std::forward<As>(as)...);
  }

protected:
  template <class Tag, class... As> void pass(Tag, As &&...as) noexcept {
    static_cast<T &>(*this).complete(Tag{}, std::forward<As>(as)...);
  }
  decltype(auto) downstream_env() const noexcept {
    return static_cast<const T &>(*this).get_env();
  }
  // Runs fn; if it may throw and does, completes with the exception instead.
  template <bool Nothrow, class Fn> bool guard(Fn &&fn) noexcept {
    return try_invoke<Nothrow>(std::forward<Fn>(fn),
      [this](auto e) noexcept { this->pass(set_error_t{}, std::move(e)); });
  }
};

namespace desc {
// The bottom layer: owns the connected receiver.
template <class R> struct receiver {
  R rcvr;
  template <class T> struct apply : sg::node::apply<T, apply<T>> {
    using base = sg::node::apply<T, apply<T>>;
    R rcvr;
    template <class... Rest>
    apply(receiver d, Rest &&...rest)
        : base{sg::node{}, std::forward<Rest>(rest)...},
          rcvr(std::move(d.rcvr)) {}
    template <class Tag, class... As> void complete(Tag, As &&...as) noexcept {
      Tag{}(std::move(rcvr), std::forward<As>(as)...);
    }
    decltype(auto) get_env() const noexcept { return exec::get_env(rcvr); }
  };
};
} // namespace desc

// Receiver for a nested operation owned by a layer. Completions go to
// target->complete(Tag, ...), or with a Key to target->child_complete(Key,
// Tag, ...). Env is spelled out because Target may still be incomplete.
template <class Target, class Env, class Key = void> struct forward_receiver {
  using receiver_concept = receiver_t;
  Target *target;
  template <class... Vs> void set_value(Vs &&...vs) && noexcept {
    deliver(set_value_t{}, std::forward<Vs>(vs)...);
  }
  template <class E> void set_error(E &&e) && noexcept {
    deliver(set_error_t{}, std::forward<E>(e));
  }
  void set_stopped() && noexcept { deliver(set_stopped_t{}); }
  Env get_env() const noexcept { return target->get_env(); }

private:
  template <class Tag, class... As> void deliver(Tag, As &&...as) noexcept {
    if constexpr (std::is_void_v<Key>)
      target->complete(Tag{}, std::forward<As>(as)...);
    else
      target->child_complete(Key{}, Tag{}, std::forward<As>(as)...);
  }
};

template <class... Ds> struct operation_graph : sg::graph<Ds...> {
  using base = sg::graph<Ds...>;
  using operation_state_concept = operation_state_t;
  using base::base;
  operation_graph(const operation_graph &) = delete;
  operation_graph(operation_graph &&) = delete;
  operation_graph &operator=(const operation_graph &) = delete;
  operation_graph &operator=(operation_graph &&) = delete;
  void start() & noexcept {
    if (std::exchange(started_, true))
      std::terminate();
    base::start(); // The outermost layer is the source.
  }

private:
  bool started_ = false;
};
template <class... Ds>
operation_graph(Ds &&...) -> operation_graph<std::remove_cvref_t<Ds>...>;

// A descriptor whose layer is told its input completion signatures.
template <class D, class In> struct bound : D {
  explicit bound(D d) : D(std::move(d)) {}
  template <class T> using apply = typename D::template apply<T, In>;
};

// The environment that the steps before D see when D sees Env. A descriptor
// that changes it defines upstream_env, and its layer defines a get_env()
// returning that type; the layers before it find it through downstream_env().
template <class D, class Env> struct upstream_env_of {
  using type = Env;
};
template <class D, class Env>
  requires requires { typename D::template upstream_env<Env>; }
struct upstream_env_of<D, Env> {
  using type = typename D::template upstream_env<Env>;
};
// The environment a step sees when Ds follow it and Env is the receiver's.
template <class Env, class... Ds> struct seen_env {
  using type = Env;
};
template <class Env, class D, class... Ds> struct seen_env<Env, D, Ds...> {
  using type =
    typename upstream_env_of<D, typename seen_env<Env, Ds...>::type>::type;
};

// Folds completion signatures from the source down to the receiver. Each
// step computes its completions in the environment it sees.
template <class Env, class In, class... Ds> struct plan {
  using sigs = In;
  using bound_list = type_list<>;
};
template <class Env, class In, class D, class... Ds>
struct plan<Env, In, D, Ds...> {
  using next = plan<Env,
    typename D::template completions<In, typename seen_env<Env, Ds...>::type>,
    Ds...>;
  using sigs = typename next::sigs;
  using bound_list = typename type_list_cat<type_list<bound<D, In>>,
    typename next::bound_list>::type;
};

template <class R, class... Ds>
auto connect_descriptors(std::tuple<Ds...> &&ds, R r) {
  using planned = plan<std::decay_t<env_of_t<R>>, type_list<>, Ds...>;
  static_assert(receiver_of<R, typename planned::sigs>,
    "connect: the receiver does not accept every completion of the sender");
  using bound_list = typename planned::bound_list;
  constexpr std::size_t n = sizeof...(Ds);
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    return operation_graph{desc::receiver<R>{std::move(r)},
      typename type_list_at<n - 1 - I, bound_list>::type(
        std::get<n - 1 - I>(std::move(ds)))...};
  }(std::index_sequence_for<Ds...>{});
}

template <class D>
concept has_attrs = requires(const D &d) { d.attrs(); };

// A sender's attributes come from its last descriptor that defines attrs();
// the others forward what precedes them.
template <class E> constexpr auto last_attrs(E e) noexcept { return e; }
template <class E, class D, class... Ds>
constexpr auto last_attrs(E e, const D &d, const Ds &...ds) noexcept {
  if constexpr (has_attrs<D>)
    return last_attrs(d.attrs(), ds...);
  else
    return last_attrs(std::move(e), ds...);
}

} // namespace detail

// A source descriptor followed by adaptor descriptors, in pipe order.
template <class... Ds> struct basic_sender {
  static_assert(sizeof...(Ds) > 0, "a sender needs a source");
  static_assert(std::tuple_element_t<0, std::tuple<Ds...>>::is_source,
    "the first descriptor must be a source");
  static_assert((int(Ds::is_source) + ...) == 1,
    "a sender must contain exactly one source");

  using sender_concept = sender_t;
  std::tuple<Ds...> descriptors;

  explicit basic_sender(Ds... ds) : descriptors(std::move(ds)...) {}

  template <class Env = env<>>
  constexpr auto get_completion_signatures(Env && = {}) const noexcept ->
    typename detail::plan<std::decay_t<Env>, type_list<>, Ds...>::sigs {
    return {};
  }

  constexpr auto get_env() const noexcept {
    return std::apply(
      [](const Ds &...ds) noexcept {
        return detail::last_attrs(env<>{}, ds...);
      },
      descriptors);
  }

  template <receiver R> auto connect(R r) && {
    return detail::connect_descriptors(std::move(descriptors), std::move(r));
  }
  template <receiver R>
    requires(std::copy_constructible<Ds> && ...)
  auto connect(R r) const & {
    return detail::connect_descriptors(
      std::tuple<Ds...>(descriptors), std::move(r));
  }
};
template <class... Ds> basic_sender(Ds...) -> basic_sender<Ds...>;

namespace detail {
template <class S> inline constexpr bool is_basic_sender = false;
template <class... Ds>
inline constexpr bool is_basic_sender<basic_sender<Ds...>> = true;

namespace desc {
// Connects a sender that is not a basic_sender as a nested operation.
template <class S> struct nested {
  S sndr;
  static constexpr bool is_source = true;
  template <class, class Env>
  using completions = completion_signatures_of_t<S, Env>;
  auto attrs() const noexcept { return exec::get_env(sndr); }
  template <class T, class In> struct apply : layer<T, apply<T, In>> {
    using base = layer<T, apply<T, In>>;
    using receiver_type = forward_receiver<apply, env_of_t<const T &>>;
    connect_result_t<S, receiver_type> op;
    template <class... Rest>
    apply(nested d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...},
          op(exec::connect(std::move(d.sndr), receiver_type{this})) {}
    void start() noexcept { exec::start(op); }
  };
};
} // namespace desc

// Appends descriptor d to sender s.
template <class S, class D> auto append(S &&s, D d) {
  if constexpr (is_basic_sender<std::remove_cvref_t<S>>)
    return std::apply(
      [&](auto... ds) { return basic_sender{std::move(ds)..., std::move(d)}; },
      std::forward<S>(s).descriptors);
  else
    return basic_sender{
      desc::nested<std::remove_cvref_t<S>>{std::forward<S>(s)}, std::move(d)};
}
} // namespace detail

template <class D> struct sender_adaptor_closure {};

namespace detail {
template <class C>
concept adaptor_closure = std::derived_from<std::remove_cvref_t<C>,
  sender_adaptor_closure<std::remove_cvref_t<C>>>;

// Calls fn(s, args...) on the sender s it is applied to.
template <class Fn, class... As>
struct bind_back : sender_adaptor_closure<bind_back<Fn, As...>> {
  [[no_unique_address]] Fn fn;
  std::tuple<As...> args;
  bind_back(Fn f, As... as) : fn(f), args(std::move(as)...) {}
  template <sender S> auto operator()(S &&s) && {
    return std::apply(
      [&](As &...as) { return fn(std::forward<S>(s), std::move(as)...); },
      args);
  }
  template <sender S> auto operator()(S &&s) const & {
    return std::apply(
      [&](const As &...as) { return fn(std::forward<S>(s), as...); }, args);
  }
};

template <class... As> inline constexpr bool leads_with_sender = false;
template <class A, class... As>
inline constexpr bool leads_with_sender<A, As...> = sender<A>;

// Base of a pipeable adaptor object. Self takes the sender first; called
// without one, the object returns a closure that applies Self to the sender
// it is given later.
template <class Self> struct adaptor {
  template <class... As>
    requires(!leads_with_sender<As...>)
  auto operator()(As &&...as) const {
    return bind_back{Self{}, std::decay_t<As>(std::forward<As>(as))...};
  }
};

template <class A, class B>
struct composed : sender_adaptor_closure<composed<A, B>> {
  A first;
  B second;
  template <sender S> auto operator()(S &&s) && {
    return std::move(second)(std::move(first)(std::forward<S>(s)));
  }
  template <sender S> auto operator()(S &&s) const & {
    return second(first(std::forward<S>(s)));
  }
};
} // namespace detail

template <sender S, detail::adaptor_closure C> auto operator|(S &&s, C &&c) {
  return std::forward<C>(c)(std::forward<S>(s));
}
template <detail::adaptor_closure A, detail::adaptor_closure B>
auto operator|(A &&a, B &&b) {
  return detail::composed<std::remove_cvref_t<A>, std::remove_cvref_t<B>>{
    {}, std::forward<A>(a), std::forward<B>(b)};
}

// ---- Sources: just, just_error, just_stopped, read_env -------------------

namespace detail {
namespace desc {
template <class Tag, class... Vs> struct just {
  std::tuple<Vs...> values;
  static constexpr bool is_source = true;
  template <class, class> using completions = completion_signatures<Tag(Vs...)>;
  template <class T, class In> struct apply : layer<T, apply<T, In>> {
    using base = layer<T, apply<T, In>>;
    std::tuple<Vs...> values;
    template <class... Rest>
    apply(just d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...}, values(std::move(d.values)) {}
    void start() noexcept {
      std::apply(
        [this](Vs &...vs) noexcept { this->pass(Tag{}, std::move(vs)...); },
        values);
    }
  };
};

template <class Q> struct read_env {
  [[no_unique_address]] Q query;
  static constexpr bool is_source = true;
  template <class, class Env>
  using completions = completion_signatures<set_value_t(
    std::decay_t<std::invoke_result_t<const Q &, const Env &>>)>;
  template <class T, class In> struct apply : layer<T, apply<T, In>> {
    using base = layer<T, apply<T, In>>;
    [[no_unique_address]] Q query;
    template <class... Rest>
    apply(read_env d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...}, query(d.query) {}
    void start() noexcept {
      using value = std::decay_t<
        std::invoke_result_t<const Q &, decltype(this->downstream_env())>>;
      this->pass(set_value_t{}, value(query(this->downstream_env())));
    }
  };
};
} // namespace desc
} // namespace detail

struct just_t {
  template <class... Vs> auto operator()(Vs &&...vs) const {
    return basic_sender{detail::desc::just<set_value_t, std::decay_t<Vs>...>{
      {std::forward<Vs>(vs)...}}};
  }
};
struct just_error_t {
  template <class E> auto operator()(E &&e) const {
    return basic_sender{
      detail::desc::just<set_error_t, std::decay_t<E>>{{std::forward<E>(e)}}};
  }
};
struct just_stopped_t {
  auto operator()() const {
    return basic_sender{detail::desc::just<set_stopped_t>{{}}};
  }
};
struct read_env_t {
  template <class Q> auto operator()(Q q) const {
    return basic_sender{detail::desc::read_env<Q>{q}};
  }
};
inline constexpr just_t just{};
inline constexpr just_error_t just_error{};
inline constexpr just_stopped_t just_stopped{};
inline constexpr read_env_t read_env{};

// ---- then, upon_error, upon_stopped -------------------------------------

namespace detail {
template <class Sig, class Tag, class F> struct then_sig {
  using type = completion_signatures<Sig>;
};
template <class Tag, class F, class... As> struct then_sig<Tag(As...), Tag, F> {
  using type =
    concat_sigs_t<completion_signatures<
                    typename value_sig<std::invoke_result_t<F &, As...>>::type>,
      exception_sigs<std::is_nothrow_invocable_v<F &, As...>>>;
};

namespace desc {
template <class Tag, class F> struct then {
  F function;
  static constexpr bool is_source = false;
  template <class In, class>
  using completions =
    sig_map_t<In, type_bind_back<then_sig, Tag, F>::template apply>;
  template <class T, class In> struct apply : layer<T, apply<T, In>> {
    using base = layer<T, apply<T, In>>;
    F function;
    template <class... Rest>
    apply(then d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...}, function(std::move(d.function)) {}
    template <class C, class... As> void complete(C, As &&...as) noexcept {
      if constexpr (!std::is_same_v<C, Tag>) {
        this->pass(C{}, std::forward<As>(as)...);
      } else {
        this->template guard<std::is_nothrow_invocable_v<F &, As...>>([&] {
          if constexpr (std::is_void_v<std::invoke_result_t<F &, As...>>) {
            std::invoke(function, std::forward<As>(as)...);
            this->pass(set_value_t{});
          } else {
            this->pass(
              set_value_t{}, std::invoke(function, std::forward<As>(as)...));
          }
        });
      }
    }
  };
};
} // namespace desc

// ---- let_value, let_error, let_stopped ----------------------------------

// Inputs are kept alive next to the operation they produce.
template <class Rcvr, class F, class... Vs> struct let_state {
  std::tuple<Vs...> args;
  connect_result_t<std::invoke_result_t<F &, Vs &...>, Rcvr> op;
  template <class... As>
  let_state(F &f, Rcvr r, As &&...as)
      : args(std::forward<As>(as)...),
        op(exec::connect(std::apply(f, args), std::move(r))) {}
};

template <class Sig, class Tag, class F, class Env> struct let_sig {
  using type = completion_signatures<Sig>;
};
template <class Tag, class F, class Env, class... As>
struct let_sig<Tag(As...), Tag, F, Env> {
  using child = std::invoke_result_t<F &, std::decay_t<As> &...>;
  static_assert(sender<child>, "let_* function must return a sender");
  using type = concat_sigs_t<completion_signatures_of_t<child, Env>,
    exception_sigs<false>>;
};

namespace desc {
template <class Tag, class F> struct let {
  F function;
  static constexpr bool is_source = false;
  template <class In, class Env>
  using completions =
    sig_map_t<In, type_bind_back<let_sig, Tag, F, Env>::template apply>;
  // Where the child sender completes is not known here.
  env<> attrs() const noexcept { return {}; }
  template <class T, class In> struct apply : layer<T, apply<T, In>> {
    using base = layer<T, apply<T, In>>;
    using receiver_type = forward_receiver<T, env_of_t<const T &>>;
    template <class... As>
    using state = let_state<receiver_type, F, std::decay_t<As>...>;
    F function;
    gather_signatures_t<Tag, In, state, monostate_variant> ops;

    template <class... Rest>
    apply(let d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...}, function(std::move(d.function)) {}

    template <class C, class... As> void complete(C, As &&...as) noexcept {
      if constexpr (!std::is_same_v<C, Tag>) {
        this->pass(C{}, std::forward<As>(as)...);
      } else {
        state<As...> *s = nullptr;
        if (this->template guard<false>([&] {
              s = &ops.template emplace<state<As...>>(
                function, receiver_type{this}, std::forward<As>(as)...);
            }))
          exec::start(s->op); // May complete and destroy this graph.
      }
    }
  };
};
} // namespace desc
} // namespace detail

namespace detail {
// then, upon_*, let_*: appends D<Tag, F> to the sender.
template <template <class, class> class D, class Tag>
struct function_adaptor : adaptor<function_adaptor<D, Tag>> {
  using adaptor<function_adaptor>::operator();
  template <sender S, class F> auto operator()(S &&s, F f) const {
    return append(std::forward<S>(s), D<Tag, F>{std::move(f)});
  }
};
} // namespace detail

using then_t = detail::function_adaptor<detail::desc::then, set_value_t>;
using upon_error_t = detail::function_adaptor<detail::desc::then, set_error_t>;
using upon_stopped_t =
  detail::function_adaptor<detail::desc::then, set_stopped_t>;
using let_value_t = detail::function_adaptor<detail::desc::let, set_value_t>;
using let_error_t = detail::function_adaptor<detail::desc::let, set_error_t>;
using let_stopped_t =
  detail::function_adaptor<detail::desc::let, set_stopped_t>;
inline constexpr then_t then{};
inline constexpr upon_error_t upon_error{};
inline constexpr upon_stopped_t upon_stopped{};
inline constexpr let_value_t let_value{};
inline constexpr let_error_t let_error{};
inline constexpr let_stopped_t let_stopped{};

// ---- write_env ----------------------------------------------------------

namespace detail {
namespace desc {
// A graph state, not a step: it holds E, and the layers before it see E in
// front of the environment below. Completions pass through it by
// inheritance.
template <class E> struct write_env {
  E environment;
  static constexpr bool is_source = false;
  template <class In, class> using completions = In;
  template <class Env> using upstream_env = env<E, Env>;
  template <class T, class In>
  struct apply : sg::state::apply<T, apply<T, In>> {
    using base = sg::state::apply<T, apply<T, In>>;
    using outer_env = std::decay_t<env_of_t<const T &>>;
    E environment;
    template <class... Rest>
    apply(write_env d, Rest &&...rest)
        : base{sg::state{}, std::forward<Rest>(rest)...},
          environment(std::move(d.environment)) {}
    env<E, outer_env> get_env() const noexcept {
      return {environment, static_cast<const T &>(*this).get_env()};
    }
  };
};
} // namespace desc
} // namespace detail

struct write_env_t : detail::adaptor<write_env_t> {
  using adaptor::operator();
  template <sender S, class E> auto operator()(S &&s, E e) const {
    return detail::append(
      std::forward<S>(s), detail::desc::write_env<E>{std::move(e)});
  }
};
inline constexpr write_env_t write_env{};

// ---- into_variant, stopped_as_optional, stopped_as_error -----------------

namespace detail {
template <class In>
using into_variant_type =
  gather_signatures_t<set_value_t, In, decayed_tuple, variant_or_empty>;

namespace desc {
struct into_variant {
  static constexpr bool is_source = false;
  template <class In, class>
  using completions = concat_sigs_t<
    std::conditional_t<type_list_size<gather_signatures_t<set_value_t, In,
                         type_list, type_list>>::value == 0,
      type_list<>, completion_signatures<set_value_t(into_variant_type<In>)>>,
    sig_map_t<In, non_value_sig>, exception_sigs<nothrow_decay_v<In>>>;
  template <class T, class In> struct apply : layer<T, apply<T, In>> {
    using base = layer<T, apply<T, In>>;
    template <class... Rest>
    apply(into_variant, Rest &&...rest) : base{std::forward<Rest>(rest)...} {}
    template <class C, class... As> void complete(C, As &&...as) noexcept {
      if constexpr (!std::is_same_v<C, set_value_t>) {
        this->pass(C{}, std::forward<As>(as)...);
      } else {
        this->template guard<nothrow_decay_v<In>>([&] {
          this->pass(set_value_t{},
            into_variant_type<In>(std::in_place_type<decayed_tuple<As...>>,
              std::forward<As>(as)...));
        });
      }
    }
  };
};
} // namespace desc

template <class Sig> struct as_optional_sig {
  using type = completion_signatures<Sig>;
};
template <class V> struct as_optional_sig<set_value_t(V)> {
  using type = concat_sigs_t<
    completion_signatures<set_value_t(std::optional<std::decay_t<V>>)>,
    exception_sigs<std::is_nothrow_constructible_v<std::decay_t<V>, V>>>;
};
template <> struct as_optional_sig<set_stopped_t()> {
  using type = completion_signatures<>;
};

namespace desc {
struct stopped_as_optional {
  static constexpr bool is_source = false;
  template <class In, class> using completions = sig_map_t<In, as_optional_sig>;
  template <class T, class In> struct apply : layer<T, apply<T, In>> {
    using base = layer<T, apply<T, In>>;
    template <class... Vs> using single_decayed = std::decay_t<single_t<Vs...>>;
    using values =
      gather_signatures_t<set_value_t, In, single_decayed, type_list>;
    static_assert(type_list_size<values>::value == 1,
      "stopped_as_optional requires exactly one value completion of one value");
    using optional_type = std::optional<typename type_list_at<0, values>::type>;
    template <class... Rest>
    apply(stopped_as_optional, Rest &&...rest)
        : base{std::forward<Rest>(rest)...} {}
    template <class C, class... As> void complete(C, As &&...as) noexcept {
      if constexpr (std::is_same_v<C, set_value_t>) {
        this->template guard<std::is_nothrow_constructible_v<
          typename optional_type::value_type, As...>>([&] {
          this->pass(set_value_t{},
            optional_type(std::in_place, std::forward<As>(as)...));
        });
      } else if constexpr (std::is_same_v<C, set_stopped_t>) {
        this->pass(set_value_t{}, optional_type{});
      } else {
        this->pass(C{}, std::forward<As>(as)...);
      }
    }
  };
};
} // namespace desc
} // namespace detail

struct into_variant_t : detail::adaptor<into_variant_t> {
  using adaptor::operator();
  template <sender S> auto operator()(S &&s) const {
    return detail::append(std::forward<S>(s), detail::desc::into_variant{});
  }
};
struct stopped_as_optional_t : detail::adaptor<stopped_as_optional_t> {
  using adaptor::operator();
  template <sender S> auto operator()(S &&s) const {
    return detail::append(
      std::forward<S>(s), detail::desc::stopped_as_optional{});
  }
};
struct stopped_as_error_t : detail::adaptor<stopped_as_error_t> {
  using adaptor::operator();
  template <sender S, class E> auto operator()(S &&s, E e) const {
    return let_stopped(std::forward<S>(s),
      [e = std::move(e)]() mutable { return just_error(std::move(e)); });
  }
};
inline constexpr into_variant_t into_variant{};
inline constexpr stopped_as_optional_t stopped_as_optional{};
inline constexpr stopped_as_error_t stopped_as_error{};

// ---- schedule_from, continues_on, starts_on, on --------------------------

namespace detail {
namespace desc {
// Stores the decayed completion, then completes it again from the scheduler.
template <class Sch> struct schedule_from {
  Sch sch;
  static constexpr bool is_source = false;
  sched_attrs<Sch> attrs() const noexcept { return {sch}; }
  template <class In>
  using stored_sigs = concat_sigs_t<sig_map_t<In, decay_sig>,
    exception_sigs<nothrow_decay_v<In>>>;
  template <class In, class Env>
  using completions = concat_sigs_t<stored_sigs<In>,
    sig_map_t<completion_signatures_of_t<schedule_result_t<Sch &>, Env>,
      non_value_sig>>;
  template <class T, class In> struct apply : layer<T, apply<T, In>> {
    using base = layer<T, apply<T, In>>;
    using receiver_type =
      forward_receiver<apply, env_of_t<const T &>, std::in_place_index_t<0>>;
    template <class... Ss>
    static auto variant_of(
      type_list<Ss...>) -> monostate_variant<typename sig<Ss>::tuple...>;

    Sch sch;
    decltype(variant_of(stored_sigs<In>{})) result;
    connect_result_t<schedule_result_t<Sch &>, receiver_type> op;

    template <class... Rest>
    apply(schedule_from d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...}, sch(std::move(d.sch)),
          op(exec::connect(exec::schedule(sch), receiver_type{this})) {}

    template <class C, class... As> void complete(C, As &&...as) noexcept {
      if (this->template guard<nothrow_decay_v<In>>([&] {
            result.template emplace<std::tuple<C, std::decay_t<As>...>>(
              C{}, std::forward<As>(as)...);
          }))
        exec::start(op);
    }
    // Completion of the schedule operation.
    template <class C, class... As>
    void child_complete(std::in_place_index_t<0>, C, As &&...as) noexcept {
      if constexpr (!std::is_same_v<C, set_value_t>)
        this->pass(C{}, std::forward<As>(as)...);
      else
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
          result);
    }
  };
};
} // namespace desc
} // namespace detail

struct schedule_from_t {
  template <scheduler Sch, sender S> auto operator()(Sch sch, S &&s) const {
    return detail::append(
      std::forward<S>(s), detail::desc::schedule_from<Sch>{std::move(sch)});
  }
};
inline constexpr schedule_from_t schedule_from{};

struct continues_on_t : detail::adaptor<continues_on_t> {
  using adaptor::operator();
  template <sender S, scheduler Sch> auto operator()(S &&s, Sch sch) const {
    return schedule_from(std::move(sch), std::forward<S>(s));
  }
};
inline constexpr continues_on_t continues_on{};

struct starts_on_t {
  template <scheduler Sch, sender S> auto operator()(Sch sch, S &&s) const {
    return let_value(schedule(sch),
      [sch, s = std::remove_cvref_t<S>(std::forward<S>(s))]() mutable {
        return write_env(std::move(s), prop{get_scheduler, sch});
      });
  }
};
inline constexpr starts_on_t starts_on{};

struct on_t {
  // Runs s on sch, then returns to the scheduler of the receiver's
  // environment.
  template <scheduler Sch, sender S> auto operator()(Sch sch, S &&s) const {
    return let_value(read_env(get_scheduler),
      [sch, s = std::remove_cvref_t<S>(std::forward<S>(s))](auto orig) mutable {
        return continues_on(starts_on(sch, std::move(s)), orig);
      });
  }
  // Completes s, runs closure c on sch, then returns to the original
  // scheduler.
  template <sender S, scheduler Sch, detail::adaptor_closure C>
  auto operator()(S &&s, Sch sch, C &&c) const {
    return let_value(read_env(get_scheduler),
      [sch, s = std::remove_cvref_t<S>(std::forward<S>(s)),
        c = std::remove_cvref_t<C>(std::forward<C>(c))](auto orig) mutable {
        return continues_on(
          write_env(std::move(c)(continues_on(std::move(s), sch)),
            prop{get_scheduler, sch}),
          orig);
      });
  }
  template <scheduler Sch, detail::adaptor_closure C>
  auto operator()(Sch sch, C &&c) const {
    return detail::bind_back{
      on_t{}, std::move(sch), std::remove_cvref_t<C>(std::forward<C>(c))};
  }
};
inline constexpr on_t on{};

// ---- when_all -----------------------------------------------------------

namespace detail {
template <class List> struct front_or_empty {
  using type = std::tuple<>;
};
template <class T, class... Ts> struct front_or_empty<type_list<T, Ts...>> {
  using type = T;
};

template <class Env, class... Ss> struct when_all_traits {
  using child_env = env<prop<get_stop_token_t, inplace_stop_token>, Env>;
  template <class S> using sigs_of = completion_signatures_of_t<S, child_env>;
  template <class S>
  using values_of =
    gather_signatures_t<set_value_t, sigs_of<S>, decayed_tuple, type_list>;
  static_assert(((type_list_size<values_of<Ss>>::value <= 1) && ...),
    "when_all requires at most one value completion per child");
  static constexpr bool has_values =
    ((type_list_size<values_of<Ss>>::value == 1) && ...);
  static constexpr bool nothrow = (nothrow_decay_v<sigs_of<Ss>> && ...);
  using value_tuple = decltype(std::tuple_cat(
    std::declval<typename front_or_empty<values_of<Ss>>::type>()...));
  template <class... Vs>
  static auto value_sig(std::tuple<Vs...>) -> type_list<set_value_t(Vs...)>;
  using sigs = concat_sigs_t<
    std::conditional_t<has_values,
      decltype(value_sig(std::declval<value_tuple>())), type_list<>>,
    sig_map_t<sigs_of<Ss>, error_sig>..., exception_sigs<nothrow>,
    completion_signatures<set_stopped_t()>>;
};

namespace desc {
// Waits for every child. The first error, or else a stop, wins and requests
// stop on the remaining children. A stop request from the receiver's
// environment is forwarded to the children.
template <class... Ss> struct when_all {
  std::tuple<Ss...> senders;
  static constexpr bool is_source = true;
  template <class, class Env>
  using completions = typename when_all_traits<Env, Ss...>::sigs;

  template <class T, class In> struct apply : layer<T, apply<T, In>> {
    using base = layer<T, apply<T, In>>;
    using outer_env = std::decay_t<env_of_t<const T &>>;
    using traits = when_all_traits<outer_env, Ss...>;
    using child_env = typename traits::child_env;
    enum : int { running, failed, stopped };

    template <std::size_t I>
    using receiver =
      forward_receiver<apply, child_env, std::in_place_index_t<I>>;
    template <std::size_t I> struct child {
      connect_result_t<std::tuple_element_t<I, std::tuple<Ss...>>, receiver<I>>
        op;
    };
    template <class Is> struct children;
    template <std::size_t... I>
    struct children<std::index_sequence<I...>> : child<I>... {
      children([[maybe_unused]] apply *self, std::tuple<Ss...> &&ss)
          : child<I>{exec::connect(
              std::get<I>(std::move(ss)), receiver<I>{self})}... {}
      void start() noexcept { (exec::start(child<I>::op), ...); }
    };
    // Declared first: children may hold callbacks registered on it.
    inplace_stop_source stop;
    std::optional<stop_callback_for_t<stop_token_of_t<outer_env>, forward_stop>>
      on_stop;
    std::atomic<int> state{running};
    std::atomic<std::size_t> pending{0};
    std::tuple<std::optional<typename front_or_empty<
      typename traits::template values_of<Ss>>::type>...>
      values;
    gather_signatures_t<set_error_t, typename traits::sigs, single_t,
      monostate_variant>
      errors;
    children<std::index_sequence_for<Ss...>> ops;

    template <class... Rest>
    apply(when_all d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...}, ops(this, std::move(d.senders)) {}

    // What the children see; this layer itself uses downstream_env().
    child_env get_env() const noexcept {
      return child_env{
        prop<get_stop_token_t, inplace_stop_token>{{}, stop.get_token()},
        outer_env(this->downstream_env())};
    }

    void start() noexcept {
      on_stop.emplace(
        get_stop_token(this->downstream_env()), forward_stop{&stop});
      if (stop.stop_requested()) {
        on_stop.reset();
        this->pass(set_stopped_t{});
        return;
      }
      // One extra arrival keeps the operation alive through all starts.
      pending.store(sizeof...(Ss) + 1, std::memory_order_relaxed);
      ops.start();
      arrive(); // Last access: finish may destroy the whole graph.
    }

    template <std::size_t I, class C, class... As>
    void child_complete(std::in_place_index_t<I>, C, As &&...as) noexcept {
      if constexpr (std::is_same_v<C, set_value_t>) {
        if (state.load(std::memory_order_relaxed) == running)
          try_invoke<traits::nothrow>(
            [&] { std::get<I>(values).emplace(std::forward<As>(as)...); },
            [this](auto e) noexcept { this->fail(std::move(e)); });
      } else if constexpr (std::is_same_v<C, set_error_t>) {
        fail(std::forward<As>(as)...);
      } else {
        int expected = running;
        if (state.compare_exchange_strong(
              expected, stopped, std::memory_order_acq_rel))
          stop.request_stop();
      }
      arrive();
    }

  private:
    template <class E> void fail(E &&e) noexcept {
      if (state.exchange(failed, std::memory_order_acq_rel) == failed)
        return;
      stop.request_stop();
      using error = std::decay_t<E>;
      try_invoke<std::is_nothrow_constructible_v<error, E>>(
        [&] { errors.template emplace<error>(std::forward<E>(e)); },
        [this](auto ep) noexcept {
          errors.template emplace<std::exception_ptr>(std::move(ep));
        });
    }

    void arrive() noexcept {
      if (pending.fetch_sub(1, std::memory_order_acq_rel) == 1)
        finish();
    }

    void finish() noexcept {
      on_stop.reset();
      switch (state.load(std::memory_order_relaxed)) {
      case running:
        if constexpr (traits::has_values)
          std::apply(
            [this](auto &...vs) noexcept {
              std::apply(
                [this](auto &...xs) noexcept {
                  this->pass(set_value_t{}, std::move(xs)...);
                },
                std::tuple_cat(std::apply(
                  [](auto &...x) noexcept { return std::tie(x...); }, *vs)...));
            },
            values);
        break;
      case failed:
        std::visit(
          [this](auto &e) noexcept {
            if constexpr (!std::is_same_v<std::decay_t<decltype(e)>,
                            std::monostate>)
              this->pass(set_error_t{}, std::move(e));
          },
          errors);
        break;
      default:
        this->pass(set_stopped_t{});
      }
    }
  };
};
} // namespace desc
} // namespace detail

struct when_all_t {
  template <sender... Ss> auto operator()(Ss &&...ss) const {
    return basic_sender{detail::desc::when_all<std::remove_cvref_t<Ss>...>{
      {std::forward<Ss>(ss)...}}};
  }
};
inline constexpr when_all_t when_all{};
struct when_all_with_variant_t {
  template <sender... Ss> auto operator()(Ss &&...ss) const {
    return when_all(into_variant(std::forward<Ss>(ss))...);
  }
};
inline constexpr when_all_with_variant_t when_all_with_variant{};

// ---- Execution resources ------------------------------------------------

namespace detail::desc {
// Completes inline, on the thread that starts it. Sch is inline_scheduler,
// which is still incomplete here.
template <class Sch> struct inline_schedule {
  static constexpr bool is_source = true;
  template <class, class>
  using completions = completion_signatures<set_value_t()>;
  sched_attrs<Sch> attrs() const noexcept { return {Sch{}}; }
  template <class T, class In> struct apply : layer<T, apply<T, In>> {
    using base = layer<T, apply<T, In>>;
    template <class... Rest>
    apply(inline_schedule, Rest &&...rest)
        : base{std::forward<Rest>(rest)...} {}
    void start() noexcept { this->pass(set_value_t{}); }
  };
};
} // namespace detail::desc

// Completes schedule() inline, on the thread that starts it.
class inline_scheduler {
public:
  using scheduler_concept = scheduler_t;
  auto schedule() const noexcept {
    return basic_sender{detail::desc::inline_schedule<inline_scheduler>{}};
  }
  friend bool operator==(inline_scheduler, inline_scheduler) noexcept = default;
};

namespace detail {
struct task {
  task *next = nullptr;
  void (*execute)(task *) noexcept = nullptr;
};

class task_queue;

// Chunks [0, count) of a bulk operation as one queued task. Each worker that
// pops it claims the next chunk and queues it again while chunks remain, so
// the job fans out across the workers without a task per chunk.
struct bulk_job : task {
  std::size_t count = 0;
  void (*run_chunk)(bulk_job &, std::size_t) noexcept = nullptr;
  std::atomic<std::size_t> next{0};
  task_queue *queue = nullptr;
};

// A FIFO of intrusive tasks, drained by run() on any number of threads. After
// finish(), queued work still runs, push fails, and run() returns once the
// queue is empty.
class task_queue {
  std::mutex mutex_;
  std::condition_variable ready_;
  task *head_ = nullptr;
  task *tail_ = nullptr;
  bool finishing_ = false;

public:
  std::size_t workers = 1; // threads that run() it

  // Notify under the lock: once run() can observe the change, the owner may be
  // destroyed as soon as the mutex is released.
  bool push(task &job) noexcept {
    std::lock_guard lock(mutex_);
    if (finishing_)
      return false;
    job.next = nullptr;
    if (tail_)
      tail_->next = &job;
    else
      head_ = &job;
    tail_ = &job;
    ready_.notify_one();
    return true;
  }
  bool push(bulk_job &job) noexcept {
    job.queue = this;
    job.next.store(0, std::memory_order_relaxed);
    job.execute = [](task *t) noexcept {
      auto &job = static_cast<bulk_job &>(*t);
      auto run = job.run_chunk;
      std::size_t i = job.next.fetch_add(1, std::memory_order_relaxed);
      std::size_t end = i + 1;
      // Not queued again: no one else can claim, so run the rest here.
      if (end < job.count && !job.queue->push(static_cast<task &>(job)))
        end = job.count;
      // A claimed chunk keeps job alive until it runs; after the last, job
      // may be gone, so only locals are used.
      for (; i < end; ++i)
        run(job, i);
    };
    return push(static_cast<task &>(job));
  }
  void finish() noexcept {
    std::lock_guard lock(mutex_);
    finishing_ = true;
    ready_.notify_all();
  }
  void run() {
    for (;;) {
      task *job;
      {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [&] { return head_ || finishing_; });
        if (!head_)
          return;
        job = head_;
        head_ = job->next;
        if (!head_)
          tail_ = nullptr;
      }
      job->execute(job); // Already unlinked. Completion may destroy job.
    }
  }
};

inline std::exception_ptr closed_error() noexcept {
  try {
    throw std::runtime_error("execution resource is finishing");
  } catch (...) {
    return std::current_exception();
  }
}

namespace desc {
template <class Owner> struct queue_schedule;
}

// The scheduler of an execution resource that runs a task_queue. Owner only
// makes the type distinct.
template <class Owner> class queue_scheduler {
  task_queue *queue_;

public:
  using scheduler_concept = scheduler_t;
  explicit queue_scheduler(task_queue *queue) noexcept : queue_(queue) {}
  basic_sender<desc::queue_schedule<Owner>> schedule() const noexcept {
    return basic_sender{desc::queue_schedule<Owner>{queue_}};
  }
  constexpr forward_progress_guarantee query(
    get_forward_progress_guarantee_t) const noexcept {
    return forward_progress_guarantee::parallel;
  }
  std::size_t query(get_max_concurrency_t) const noexcept {
    return queue_->workers;
  }
  // Runs a parallel bulk's chunks; false if the queue is finishing.
  bool schedule_bulk(bulk_job &job) const noexcept { return queue_->push(job); }
  friend bool operator==(
    const queue_scheduler &, const queue_scheduler &) noexcept = default;
};

namespace desc {
// Queues a task; when it runs, completes with set_stopped if stop was
// requested, else set_value. Fails if the queue is finishing.
template <class Owner> struct queue_schedule {
  task_queue *queue;
  static constexpr bool is_source = true;
  template <class, class>
  using completions = completion_signatures<set_value_t(),
    set_error_t(std::exception_ptr), set_stopped_t()>;
  sched_attrs<queue_scheduler<Owner>> attrs() const noexcept {
    return {queue_scheduler<Owner>{queue}};
  }
  template <class T, class In> struct apply : layer<T, apply<T, In>>, task {
    using base = layer<T, apply<T, In>>;
    task_queue *queue;
    template <class... Rest>
    apply(queue_schedule d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...}, queue(d.queue) {}
    void start() noexcept {
      execute = [](task *t) noexcept {
        auto &self = *static_cast<apply *>(t);
        if (get_stop_token(self.downstream_env()).stop_requested())
          self.pass(set_stopped_t{});
        else
          self.pass(set_value_t{});
      };
      if (!queue->push(*this))
        this->pass(set_error_t{}, closed_error());
    }
  };
};
} // namespace desc
} // namespace detail

// A FIFO execution resource drained by run(). After finish(), queued work
// still runs, and new schedule operations complete with an error.
class run_loop {
  detail::task_queue queue_;

public:
  using scheduler = detail::queue_scheduler<run_loop>;
  run_loop() = default;
  run_loop(const run_loop &) = delete;
  run_loop &operator=(const run_loop &) = delete;

  scheduler get_scheduler() noexcept { return scheduler{&queue_}; }
  void run() { queue_.run(); }
  void finish() noexcept { queue_.finish(); }
};

namespace detail {
struct parallel_resource;
}
// The scheduler of the process-wide pool returned by get_parallel_scheduler().
using parallel_scheduler = detail::queue_scheduler<detail::parallel_resource>;
inline parallel_scheduler get_parallel_scheduler();

// A fixed number of threads draining one shared queue. request_stop() rejects
// new work; the destructor requests stop, runs the queued work and joins.
class static_thread_pool {
  detail::task_queue queue_;
  std::vector<std::thread> threads_;

  void join() noexcept {
    for (auto &t : threads_)
      t.join();
  }

public:
  using scheduler = detail::queue_scheduler<static_thread_pool>;
  explicit static_thread_pool(
    std::size_t threads = std::thread::hardware_concurrency()) {
    threads = std::max<std::size_t>(threads, 1);
    queue_.workers = threads;
    threads_.reserve(threads);
    try {
      while (threads_.size() < threads)
        threads_.emplace_back([this] { queue_.run(); });
    } catch (...) {
      request_stop();
      join();
      throw;
    }
  }
  static_thread_pool(const static_thread_pool &) = delete;
  static_thread_pool &operator=(const static_thread_pool &) = delete;
  ~static_thread_pool() {
    request_stop();
    join();
  }

  scheduler get_scheduler() noexcept { return scheduler{&queue_}; }
  void request_stop() noexcept { queue_.finish(); }
  friend parallel_scheduler get_parallel_scheduler();
};

// A pool with one thread per hardware thread, started on first use.
inline parallel_scheduler get_parallel_scheduler() {
  static static_thread_pool pool;
  return parallel_scheduler{&pool.queue_};
}

// ---- bulk, bulk_chunked, bulk_unchunked ---------------------------------

namespace detail {
template <class P>
concept execution_policy = std::is_execution_policy_v<std::remove_cvref_t<P>>;
template <class P>
inline constexpr bool parallel_policy =
  std::is_same_v<std::remove_cvref_t<P>, std::execution::parallel_policy> ||
  std::is_same_v<std::remove_cvref_t<P>,
    std::execution::parallel_unsequenced_policy>;

// bulk with a parallel policy splits its work across these schedulers.
template <class Sch>
concept splits_bulk = requires(const Sch &sch) { get_max_concurrency(sch); };
// Those that also run the chunks themselves; the others get a schedule
// operation per chunk.
template <class Sch>
concept runs_bulk = requires(const Sch &sch, bulk_job &job) {
  { sch.schedule_bulk(job) } noexcept -> std::same_as<bool>;
};

enum class bulk_kind { each, chunked, unchunked };
struct no_scheduler {};

// Adapts the user function to run over a range [b, e).
template <bulk_kind Kind, class Shape, class F> struct bulk_function {
  F function;
  template <class... Vs>
  static constexpr bool nothrow = Kind == bulk_kind::chunked
    ? std::is_nothrow_invocable_v<F &, Shape, Shape, Vs &...>
    : std::is_nothrow_invocable_v<F &, Shape, Vs &...>;
  template <class... Vs> void operator()(Shape b, Shape e, Vs &...vs) {
    if constexpr (Kind == bulk_kind::chunked)
      std::invoke(function, b, e, vs...);
    else
      for (; b != e; ++b)
        std::invoke(function, b, vs...);
  }
};

template <class Sch, class Env> struct bulk_schedule_sigs {
  using type = concat_sigs_t<exception_sigs<false>,
    sig_map_t<completion_signatures_of_t<schedule_result_t<Sch &>, Env>,
      non_value_sig>>;
};
template <class Env> struct bulk_schedule_sigs<no_scheduler, Env> {
  using type = type_list<>;
};

namespace desc {
// Runs the function over [0, shape) on each value completion, then passes
// the values on. With a scheduler (Sch is not no_scheduler) the values are
// stored, the range is split into chunks that run on it, and the last chunk
// to finish completes. Otherwise the range runs inline.
template <bulk_kind Kind, class Shape, class F, class Sch> struct bulk {
  using fn = bulk_function<Kind, Shape, F>;
  static constexpr bool parallel = !std::is_same_v<Sch, no_scheduler>;
  template <class A>
  using arg_t =
    std::conditional_t<parallel, std::decay_t<A>, std::remove_reference_t<A>>;

  Shape shape;
  fn function;
  [[no_unique_address]] Sch sch;
  static constexpr bool is_source = false;

  template <class Sig> struct sig_of {
    using type = completion_signatures<Sig>;
  };
  template <class... As> struct sig_of<set_value_t(As...)> {
    using type = concat_sigs_t<completion_signatures<set_value_t(arg_t<As>...)>,
      exception_sigs<fn::template nothrow<arg_t<As>...>>>;
  };
  template <class In, class Env>
  using completions = concat_sigs_t<sig_map_t<In, sig_of>,
    typename bulk_schedule_sigs<Sch, Env>::type>;

  template <class T, class In>
  struct inline_layer : layer<T, inline_layer<T, In>> {
    using base = layer<T, inline_layer<T, In>>;
    Shape shape;
    fn function;
    template <class... Rest>
    inline_layer(bulk d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...}, shape(d.shape),
          function(std::move(d.function)) {}
    template <class C, class... As> void complete(C, As &&...as) noexcept {
      if constexpr (!std::is_same_v<C, set_value_t>)
        this->pass(C{}, std::forward<As>(as)...);
      else if (this->template guard<fn::template nothrow<arg_t<As>...>>(
                 [&] { function(Shape(0), shape, as...); }))
        this->pass(set_value_t{}, std::forward<As>(as)...);
    }
  };

  template <class T, class In>
  struct parallel_layer : layer<T, parallel_layer<T, In>> {
    using base = layer<T, parallel_layer<T, In>>;
    using outer_env = env_of_t<const T &>;
    enum : int { running, failed, stopped };

    struct chunk_receiver {
      using receiver_concept = receiver_t;
      parallel_layer *self;
      std::size_t index;
      void set_value() && noexcept { self->run_chunk(index); }
      template <class E> void set_error(E &&e) && noexcept {
        self->fail(std::forward<E>(e));
        self->arrive();
      }
      void set_stopped() && noexcept { self->stop_chunk(); }
      outer_env get_env() const noexcept { return self->downstream_env(); }
    };
    struct slot {
      connect_result_t<schedule_result_t<Sch &>, chunk_receiver> op;
      slot(Sch &sch, chunk_receiver r)
          : op(exec::connect(exec::schedule(sch), r)) {}
    };
    struct job : bulk_job {
      parallel_layer *self;
    };
    using chunks_type =
      std::conditional_t<runs_bulk<Sch>, job, std::vector<std::optional<slot>>>;
    using errors_type = gather_signatures_t<set_error_t,
      concat_sigs_t<
        sig_map_t<completion_signatures_of_t<schedule_result_t<Sch &>,
                    std::decay_t<outer_env>>,
          error_sig>,
        exception_sigs<false>>,
      single_t, monostate_variant>;

    Sch sch;
    Shape shape;
    fn function;
    std::size_t chunks = 0;
    std::atomic<int> state{running};
    std::atomic<std::size_t> pending{0};
    gather_signatures_t<set_value_t, In, decayed_tuple, monostate_variant>
      values;
    errors_type errors;
    chunks_type tasks;

    template <class... Rest>
    parallel_layer(bulk d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...}, sch(std::move(d.sch)),
          shape(d.shape), function(std::move(d.function)) {
      if constexpr (runs_bulk<Sch>) {
        tasks.self = this;
        tasks.run_chunk = [](bulk_job &j, std::size_t i) noexcept {
          auto &self = *static_cast<job &>(j).self;
          if (get_stop_token(self.downstream_env()).stop_requested())
            self.stop_chunk();
          else
            self.run_chunk(i);
        };
      }
    }

    template <class C, class... As> void complete(C, As &&...as) noexcept {
      if constexpr (!std::is_same_v<C, set_value_t>) {
        this->pass(C{}, std::forward<As>(as)...);
      } else {
        if (!this->template guard<false>([&] {
              values.template emplace<decayed_tuple<As...>>(
                std::forward<As>(as)...);
              chunks = chunk_count();
              if constexpr (!runs_bulk<Sch>)
                if (chunks > 1) {
                  tasks = std::vector<std::optional<slot>>(chunks);
                  for (std::size_t i = 0; i < chunks; ++i)
                    tasks[i].emplace(sch, chunk_receiver{this, i});
                }
            }))
          return;
        if (chunks <= 1) {
          run_range(Shape(0), shape);
          finish();
          return;
        }
        // One extra arrival keeps the operation alive through all starts.
        pending.store(chunks + 1, std::memory_order_relaxed);
        if constexpr (runs_bulk<Sch>) {
          tasks.count = chunks;
          if (!sch.schedule_bulk(tasks)) { // No chunk will run.
            fail(closed_error());
            pending.store(1, std::memory_order_relaxed);
          }
        } else {
          for (auto &s : tasks)
            exec::start(s->op);
        }
        arrive(); // Last access: finish may destroy the whole graph.
      }
    }

  private:
    std::size_t chunk_count() const noexcept {
      if (shape <= Shape(0))
        return 0;
      auto n = static_cast<std::size_t>(shape);
      if constexpr (Kind == bulk_kind::unchunked)
        return n;
      else
        return std::min(n, get_max_concurrency(sch));
    }

    template <class Fn> void visit_values(Fn &&fn) noexcept {
      std::visit(
        [&](auto &v) noexcept {
          if constexpr (!std::is_same_v<std::decay_t<decltype(v)>,
                          std::monostate>)
            std::apply(fn, v);
        },
        values);
    }

    void run_range(Shape b, Shape e) noexcept {
      visit_values([&](auto &...vs) noexcept {
        try_invoke<
          fn::template nothrow<std::remove_reference_t<decltype(vs)>...>>(
          [&] { function(b, e, vs...); },
          [&](auto ep) noexcept { fail(std::move(ep)); });
      });
    }

    // Chunk i of [0, shape); the first shape % chunks chunks get one more.
    void run_chunk(std::size_t i) noexcept {
      if (state.load(std::memory_order_relaxed) == running) {
        auto n = static_cast<std::size_t>(shape);
        std::size_t size = n / chunks, extra = n % chunks;
        std::size_t b = i * size + std::min(i, extra);
        run_range(Shape(b), Shape(b + size + (i < extra)));
      }
      arrive();
    }

    void stop_chunk() noexcept {
      int expected = running;
      state.compare_exchange_strong(
        expected, stopped, std::memory_order_acq_rel);
      arrive();
    }

    template <class E> void fail(E &&e) noexcept {
      if (state.exchange(failed, std::memory_order_acq_rel) == failed)
        return;
      using error = std::decay_t<E>;
      try_invoke<std::is_nothrow_constructible_v<error, E>>(
        [&] { errors.template emplace<error>(std::forward<E>(e)); },
        [this](auto ep) noexcept {
          errors.template emplace<std::exception_ptr>(std::move(ep));
        });
    }

    void arrive() noexcept {
      if (pending.fetch_sub(1, std::memory_order_acq_rel) == 1)
        finish();
    }

    void finish() noexcept {
      switch (state.load(std::memory_order_relaxed)) {
      case running:
        visit_values([this](auto &...vs) noexcept {
          this->pass(set_value_t{}, std::move(vs)...);
        });
        break;
      case failed:
        std::visit(
          [this](auto &e) noexcept {
            if constexpr (!std::is_same_v<std::decay_t<decltype(e)>,
                            std::monostate>)
              this->pass(set_error_t{}, std::move(e));
          },
          errors);
        break;
      default:
        this->pass(set_stopped_t{});
      }
    }
  };

  template <class T, class In>
  using apply =
    std::conditional_t<parallel, parallel_layer<T, In>, inline_layer<T, In>>;
};
} // namespace desc

// The scheduler a parallel bulk runs on: where s completes, if that resource
// is parallel and P is a parallel policy.
template <class P, class S> auto bulk_scheduler_of(const S &s) noexcept {
  if constexpr (parallel_policy<P> &&
    requires { get_completion_scheduler<set_value_t>(exec::get_env(s)); }) {
    auto sch = get_completion_scheduler<set_value_t>(exec::get_env(s));
    if constexpr (splits_bulk<decltype(sch)>)
      return sch;
    else
      return no_scheduler{};
  } else {
    return no_scheduler{};
  }
}

template <bulk_kind Kind> struct bulk_adaptor : adaptor<bulk_adaptor<Kind>> {
  using adaptor<bulk_adaptor>::operator();
  template <sender S, execution_policy P, std::integral Shape, class F>
  auto operator()(S &&s, P &&, Shape shape, F f) const {
    auto sch = bulk_scheduler_of<P>(s);
    return append(std::forward<S>(s),
      desc::bulk<Kind, Shape, F, decltype(sch)>{
        shape, {std::move(f)}, std::move(sch)});
  }
};
} // namespace detail

using bulk_t = detail::bulk_adaptor<detail::bulk_kind::each>;
using bulk_chunked_t = detail::bulk_adaptor<detail::bulk_kind::chunked>;
using bulk_unchunked_t = detail::bulk_adaptor<detail::bulk_kind::unchunked>;
inline constexpr bulk_t bulk{};
inline constexpr bulk_chunked_t bulk_chunked{};
inline constexpr bulk_unchunked_t bulk_unchunked{};

// ---- Async scopes -------------------------------------------------------

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

// ---- sync_wait ----------------------------------------------------------

namespace detail {
struct sync_wait_env {
  run_loop *loop;
  auto query(get_scheduler_t) const noexcept { return loop->get_scheduler(); }
  auto query(get_delegation_scheduler_t) const noexcept {
    return loop->get_scheduler();
  }
};

template <class Tuple> struct sync_wait_state {
  run_loop loop;
  std::optional<Tuple> values;
  std::exception_ptr error;
};

template <class E> std::exception_ptr as_exception_ptr(E &&e) noexcept {
  if constexpr (std::is_same_v<std::decay_t<E>, std::exception_ptr>)
    return std::forward<E>(e);
  else if constexpr (std::is_same_v<std::decay_t<E>, std::error_code>)
    return std::make_exception_ptr(std::system_error(e));
  else
    return std::make_exception_ptr(std::forward<E>(e));
}

template <class Tuple> struct sync_wait_receiver {
  using receiver_concept = receiver_t;
  sync_wait_state<Tuple> *state;
  template <class... As> void set_value(As &&...as) && noexcept {
    try {
      state->values.emplace(std::forward<As>(as)...);
    } catch (...) {
      state->error = std::current_exception();
    }
    state->loop.finish();
  }
  template <class E> void set_error(E &&e) && noexcept {
    state->error = as_exception_ptr(std::forward<E>(e));
    state->loop.finish();
  }
  void set_stopped() && noexcept { state->loop.finish(); }
  sync_wait_env get_env() const noexcept { return {&state->loop}; }
};
} // namespace detail

// Blocks until s completes, driving a local run_loop. Returns the values, or
// nullopt if s was stopped; rethrows an error.
struct sync_wait_t {
  template <sender_in<detail::sync_wait_env> S> auto operator()(S &&s) const {
    using values =
      value_types_of_t<S, detail::sync_wait_env, decayed_tuple, type_list>;
    static_assert(type_list_size<values>::value == 1,
      "sync_wait requires exactly one value completion");
    using tuple_type = typename type_list_at<0, values>::type;
    detail::sync_wait_state<tuple_type> state;
    auto op = exec::connect(
      std::forward<S>(s), detail::sync_wait_receiver<tuple_type>{&state});
    exec::start(op);
    state.loop.run();
    if (state.error)
      std::rethrow_exception(state.error);
    return std::move(state.values);
  }
};
struct sync_wait_with_variant_t {
  template <sender S> auto operator()(S &&s) const {
    return sync_wait_t{}(into_variant(std::forward<S>(s)));
  }
};
inline constexpr sync_wait_t sync_wait{};
inline constexpr sync_wait_with_variant_t sync_wait_with_variant{};

namespace this_thread {
using exec::sync_wait;
using exec::sync_wait_with_variant;
} // namespace this_thread

// ---- Coroutines ---------------------------------------------------------

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

// ---- task_scheduler, affine_on ------------------------------------------

namespace detail {
// The receiving end of a task_scheduler's wrapped schedule operation.
struct schedule_sink {
  inplace_stop_token token;
  virtual void set_value() noexcept = 0;
  virtual void set_error(std::exception_ptr) noexcept = 0;
  virtual void set_stopped() noexcept = 0;

protected:
  ~schedule_sink() = default;
};

struct sink_receiver {
  using receiver_concept = receiver_t;
  schedule_sink *sink;
  void set_value() && noexcept { sink->set_value(); }
  template <class E> void set_error(E &&e) && noexcept {
    sink->set_error(as_exception_ptr(std::forward<E>(e)));
  }
  void set_stopped() && noexcept { sink->set_stopped(); }
  prop<get_stop_token_t, inplace_stop_token> get_env() const noexcept {
    return {{}, sink->token};
  }
};

struct erased_operation {
  virtual ~erased_operation() = default;
  virtual void start() noexcept = 0;
};

namespace desc {
// Connects the wrapped scheduler's schedule sender on the heap. A stop token
// of the receiver reaches it if it is an inplace_stop_token. Sch is
// task_scheduler, which is still incomplete here.
template <class Sch> struct task_schedule {
  Sch sch;
  static constexpr bool is_source = true;
  template <class, class>
  using completions = completion_signatures<set_value_t(),
    set_error_t(std::exception_ptr), set_stopped_t()>;
  sched_attrs<Sch> attrs() const noexcept { return {sch}; }
  template <class T, class In> struct apply : layer<T, apply<T, In>> {
    using base = layer<T, apply<T, In>>;
    struct relay final : schedule_sink {
      apply *self;
      explicit relay(apply *a) noexcept : self(a) {}
      void set_value() noexcept override { self->pass(set_value_t{}); }
      void set_error(std::exception_ptr e) noexcept override {
        self->pass(set_error_t{}, std::move(e));
      }
      void set_stopped() noexcept override { self->pass(set_stopped_t{}); }
    };
    relay sink;
    std::unique_ptr<erased_operation> op;
    template <class... Rest>
    apply(task_schedule d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...}, sink(this),
          op(d.sch.connect(sink)) {}
    void start() noexcept {
      using token = stop_token_of_t<decltype(this->downstream_env())>;
      if constexpr (std::is_same_v<token, inplace_stop_token>)
        sink.token = get_stop_token(this->downstream_env());
      op->start();
    }
  };
};
} // namespace desc
} // namespace detail

// A scheduler of any type behind one type, shared by copies. Each schedule
// operation connects the wrapped scheduler's on the heap.
class task_scheduler {
  struct model_base {
    virtual ~model_base() = default;
    virtual std::unique_ptr<detail::erased_operation> connect(
      detail::schedule_sink &) const = 0;
    virtual bool equals(const model_base &) const noexcept = 0;
  };
  template <class Sch> struct model final : model_base {
    Sch sch;
    struct operation final : detail::erased_operation {
      connect_result_t<schedule_result_t<const Sch &>, detail::sink_receiver>
        op;
      operation(const Sch &sch, detail::schedule_sink &sink)
          : op(exec::connect(
              exec::schedule(sch), detail::sink_receiver{&sink})) {}
      void start() noexcept override { exec::start(op); }
    };
    explicit model(Sch s) : sch(std::move(s)) {}
    std::unique_ptr<detail::erased_operation> connect(
      detail::schedule_sink &sink) const override {
      return std::make_unique<operation>(sch, sink);
    }
    bool equals(const model_base &other) const noexcept override {
      auto *m = dynamic_cast<const model *>(&other);
      return m && m->sch == sch;
    }
  };

  std::shared_ptr<const model_base> sch_;
  template <class> friend struct detail::desc::task_schedule;
  std::unique_ptr<detail::erased_operation> connect(
    detail::schedule_sink &sink) const {
    return sch_->connect(sink);
  }

public:
  using scheduler_concept = scheduler_t;
  template <class Sch>
    requires(!std::is_same_v<std::remove_cvref_t<Sch>, task_scheduler> &&
      scheduler<Sch>)
  explicit task_scheduler(Sch &&sch)
      : sch_(std::make_shared<const model<std::remove_cvref_t<Sch>>>(
          std::forward<Sch>(sch))) {}
  auto schedule() const noexcept {
    return basic_sender{detail::desc::task_schedule<task_scheduler>{*this}};
  }
  friend bool operator==(
    const task_scheduler &a, const task_scheduler &b) noexcept {
    return a.sch_ == b.sch_ || a.sch_->equals(*b.sch_);
  }
};

// Completes s on sch. inline_scheduler leaves s as it is.
struct affine_on_t : detail::adaptor<affine_on_t> {
  using adaptor::operator();
  template <sender S, scheduler Sch> auto operator()(S &&s, Sch sch) const {
    if constexpr (std::is_same_v<Sch, inline_scheduler>)
      return std::remove_cvref_t<S>(std::forward<S>(s));
    else
      return continues_on(std::forward<S>(s), std::move(sch));
  }
};
inline constexpr affine_on_t affine_on{};

// ---- task ---------------------------------------------------------------

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
