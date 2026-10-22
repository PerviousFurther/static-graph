#pragma once

// sg::exec: let_value, let_error, let_stopped.

#include "../core/graph.hpp"
#include "then.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace sg::exec {

namespace detail {
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

} // namespace sg::exec
