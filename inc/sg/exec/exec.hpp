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

#include "tags.hpp"
#include "stop_token.hpp"
#include "env.hpp"
#include "awaitable.hpp"
#include "completion_signatures.hpp"
#include "concepts.hpp"
#include "layers.hpp"
#include "just.hpp"
#include "then.hpp"
#include "let.hpp"
#include "write_env.hpp"
#include "into_variant.hpp"
#include "continues_on.hpp"
#include "when_all.hpp"
#include "schedulers.hpp"
#include "bulk.hpp"
#include "scope.hpp"
#include "sync_wait.hpp"
#include "coroutine.hpp"
#include "task_scheduler.hpp"
#include "task.hpp"
