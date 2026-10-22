#pragma once

// sg::exec: Stop tokens.

#include <atomic>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <utility>

namespace sg::exec {

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

} // namespace sg::exec
