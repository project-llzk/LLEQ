/**
 * Copyright 2025--2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Analysis/Unification.h"

#include "Analysis/SymbolExpr.h"
#include "Analysis/SymbolImpls.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>

namespace lleq {

// Determine if an Unknown symbol with number `u` occurs within or is `s`
bool _occurs(unsigned u, Symbol s) {
  return llvm::TypeSwitch<Symbol, bool>(s)
      .Case<Unknown>([u](Unknown *s) { return s->n == u; })
      .Case<OpCall>([u](OpCall *s) {
        for (auto arg : s->arguments) {
          if (_occurs(u, arg)) {
            return true;
          }
        }
        return false;
      })
      .Case<Index>([u](Index *s) {
        for (auto idx : s->indices) {
          if (_occurs(u, idx)) {
            return true;
          }
        }
        return false;
      })
      .Default([](auto) { return false; });
}

llvm::LogicalResult unify(Symbol a, Symbol b, Substitutions &s) {
  // Check the easy case first
  if (*a == *b) {
    return llvm::success();
  }

  // Different constructors can't be unified
  using SymbolKind = impl::SymbolBase::SymbolKind;
  if (!llvm::isa<Unknown>(a) && !llvm::isa<Unknown>(b) && a->kind != b->kind) {
    return llvm::failure();
  }

  // Here we know that either at least one of them is unknown, or they have the
  // same constructor and we'll have to recurse

  // If one of them is unknown, make sure its not coinductive, update the
  // mapping, and succeed
  if (auto unknownA = llvm::dyn_cast<Unknown>(a)) {
    if (_occurs(unknownA->n, b)) {
      return llvm::failure();
    }
    s.push_back({unknownA->n, b});
    return llvm::success();
  }
  // By symmetry
  if (llvm::isa<Unknown>(b)) {
    return unify(b, a, s);
  }

  // Neither is unknown, so we have to recurse
  return llvm::TypeSwitch<Symbol, llvm::LogicalResult>(a)
      .Case<OpCall>([b, &s](OpCall *callA) {
        auto callB = llvm::dyn_cast<OpCall>(b);
        if (callA->opName != callB->opName) {
          return llvm::failure();
        }
        return unify_all(callA->arguments, callB->arguments, s);
      })
      .Case<Index>([b, &s](Index *idxA) {
        auto idxB = llvm::dyn_cast<Index>(b);
        if (idxA->signal != idxB->signal) {
          return llvm::failure();
        }
        return unify_all(idxA->indices, idxB->indices, s);
      })
      .Default([](auto) { return llvm::failure(); });
}

llvm::LogicalResult unify_all(llvm::ArrayRef<Symbol> as,
                              llvm::ArrayRef<Symbol> bs, Substitutions &s) {

  if (as.size() != bs.size()) {
    return llvm::failure();
  }

  for (auto [a, b] : llvm::zip(as, bs)) {
    if (llvm::failed(unify(substitute(a, s), substitute(b, s), s))) {
      return llvm::failure();
    }
  }
  return llvm::success();
}

// Substitutes Unknown symbol with id `k` within `original` with `v` (or
// replaces `original` with `v` if `original` is an unknown symbol with id `k`)
Symbol _single_subst(Symbol original, unsigned k, Symbol v) {
  return llvm::TypeSwitch<Symbol, Symbol>(original)
      .Case<Unknown>([k, v](Unknown *u) -> Symbol {
        if (u->n == k) {
          return v;
        }
        return u;
      })
      .Case<OpCall>([k, v](OpCall *c) -> Symbol {
        llvm::SmallVector<Symbol> substArgs;
        for (auto arg : c->arguments) {
          substArgs.push_back(_single_subst(arg, k, v));
        }
        return c->pool.func_call(c->opName, substArgs);
      })
      .Case<Index>([k, v](Index *i) -> Symbol {
        llvm::SmallVector<Symbol> indices;
        for (auto idx : i->indices) {
          indices.push_back(_single_subst(idx, k, v));
        }
        return i->pool.index(i->signal, indices);
      })
      .Default([original](auto) { return original; });
}

Symbol substitute(Symbol original, const Substitutions &m) {
  // To avoid filling the pool up with a bunch of temporary intermediate
  // symbols, allocate a local pool for the temporaries...
  SymbolPool local_pool;
  // ...and copy the symbol into it
  Symbol orig = local_pool.copy(original);
  for (auto [k, v] : m) {
    // _single_subst will continue returning symbols owned by `local_pool`
    orig = _single_subst(orig, k, v);
  }
  // Finally copy the result back into the original pool. When `local_pool` goes
  // out of scope, all the temporaries are automatically freed
  return original->pool.copy(orig);
}

// "Erase" whatever's different between `a` and `b` and replace it with unknowns
// Return the anti-unified symbol
static Symbol _anti_unify_impl(Symbol a, Symbol b, AUTag tag) {
  // If either is uninitialized, return the other
  if (llvm::isa<Uninitialized>(a)) {
    return b;
  }
  if (llvm::isa<Uninitialized>(b)) {
    return a;
  }
  // If they're the same, or one of them is unknown, there's no work to do
  if (impl::equal(a, b) || llvm::isa<Unknown>(a)) {
    return a;
  }
  if (llvm::isa<Unknown>(b)) {
    return b;
  }
  // If the constructors don't match there's no hope of unifying anything
  if (a->kind != b->kind) {
    return a->pool.fresh_unknown();
  }
  return llvm::TypeSwitch<Symbol, Symbol>(a)
      .Case<OpCall>([b, tag](OpCall *callA) -> Symbol {
        // If its two calls to the same function, try anti-unifying each
        // argument
        auto callB = llvm::dyn_cast<OpCall>(b);
        if (callA->opName != callB->opName ||
            callA->arguments.size() != callB->arguments.size()) {
          return callA->pool.fresh_unknown();
        }

        llvm::SmallVector<Symbol> anti_unified_args;
        for (unsigned i = 0; i < callA->arguments.size(); i++) {
          anti_unified_args.push_back(
              anti_unify(callA->arguments[i], callB->arguments[i], tag));
        }
        return callA->pool.func_call(callA->opName, anti_unified_args);
      })
      .Case<Index>([b, tag](Index *indexA) -> Symbol {
        // If its two indices into the same array, try anti-unifying each index
        auto indexB = llvm::dyn_cast<Index>(b);
        if (indexA->signal != indexB->signal ||
            indexA->indices.size() != indexB->indices.size()) {
          return indexA->pool.fresh_unknown();
        }

        llvm::SmallVector<Symbol> anti_unified_idx;
        for (unsigned i = 0; i < indexA->indices.size(); i++) {
          anti_unified_idx.push_back(
              anti_unify(indexA->indices[i], indexB->indices[i], tag));
        }
        return indexA->pool.index(indexA->signal, anti_unified_idx);
      })
      .Default([b](auto) { return b->pool.fresh_unknown(); });
}

Symbol anti_unify(Symbol a, Symbol b, AUTag tag) {
  static llvm::DenseMap<std::tuple<Symbol, Symbol, AUTag>, Symbol>
      previousLookups;

  if (tag.isNull()) {
    return _anti_unify_impl(a, b, tag);
  }
  if (previousLookups.contains({a, b, tag})) {
    return previousLookups.at({a, b, tag});
  }

  auto ans = _anti_unify_impl(a, b, tag);
  previousLookups[{a, b, tag}] = ans;
  return ans;
}

} // namespace lleq
