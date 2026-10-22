#pragma once

// sg::exec: Completion signatures.

#include "../core/graph.hpp"
#include "awaitable.hpp"

#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace sg::exec {

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

} // namespace sg::exec
