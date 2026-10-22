#pragma once

// sg::exec: then, upon_error, upon_stopped.

#include "../core/graph.hpp"
#include "layers.hpp"

#include <functional>
#include <type_traits>
#include <utility>

namespace sg::exec {

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
} // namespace detail

} // namespace sg::exec
