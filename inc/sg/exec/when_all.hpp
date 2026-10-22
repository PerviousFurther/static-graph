#pragma once

// sg::exec: when_all.

#include "../core/graph.hpp"
#include "into_variant.hpp"

#include <atomic>
#include <cstddef>
#include <exception>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace sg::exec {

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

} // namespace sg::exec
