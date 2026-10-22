#pragma once

// sg::exec: Graph layers.

#include "../core/graph.hpp"
#include "concepts.hpp"

#include <concepts>
#include <cstddef>
#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>

namespace sg::exec {

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

} // namespace sg::exec
