#pragma once

// sg::exec: into_variant, stopped_as_optional, stopped_as_error.

#include "../core/graph.hpp"
#include "just.hpp"
#include "let.hpp"

#include <optional>
#include <type_traits>
#include <utility>

namespace sg::exec {

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

} // namespace sg::exec
