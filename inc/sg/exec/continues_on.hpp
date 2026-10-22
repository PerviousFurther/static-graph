#pragma once

// sg::exec: schedule_from, continues_on, starts_on, on.

#include "../core/graph.hpp"
#include "just.hpp"
#include "let.hpp"
#include "write_env.hpp"

#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace sg::exec {

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

} // namespace sg::exec
