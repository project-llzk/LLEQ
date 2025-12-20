/**
 * Copyright 2025 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Analysis/Unification.h"
#include "Analysis/SymbolExpr.h"
#include "SymbolImpls.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Support/LLVM.h>

namespace lleq {

bool _occurs(unsigned u, Symbol s) {
  return llvm::TypeSwitch<Symbol, bool>(s)
      .Case<Unknown>([u](Unknown *s) { return s->n == u; })
      .Case<OpCall>([u](OpCall *s) {
        for (auto arg : s->arguments) {
          if (_occurs(u, arg))
            return true;
        }
        return false;
      })
      .Case<Index>([u](Index *s) {
        for (auto idx : s->indices) {
          if (_occurs(u, idx))
            return true;
        }
        return false;
      })
      .Default([](auto) { return false; });
}

mlir::LogicalResult unify(Symbol a, Symbol b, Substitutions &s) {
  // Check the easy case first
  if (*a == *b)
    return mlir::success();

  // Different constructors can't be unified
  using SymbolKind = impl::SymbolBase::SymbolKind;
  if (a->kind != SymbolKind::SK_Unknown && b->kind != SymbolKind::SK_Unknown &&
      a->kind != b->kind)
    return mlir::failure();

  // Here we know that either at least one of them is unknown, or they have the
  // same constructor and we'll have to recurse

  // If one of them is unknown, make sure its not coinductive, update the
  // mapping, and succeed
  if (a->kind == SymbolKind::SK_Unknown) {
    if (_occurs(mlir::dyn_cast<Unknown>(a)->n, b))
      return mlir::failure();
    s.push_back({mlir::dyn_cast<Unknown>(a)->n, b});
    return mlir::success();
  }
  // By symmetry
  if (b->kind == SymbolKind::SK_Unknown) {
    return unify(b, a, s);
  }

  // Neither is unknown, so we have to recurse
  return llvm::TypeSwitch<Symbol, mlir::LogicalResult>(a)
      .Case<OpCall>([b, &s](OpCall *callA) {
        auto callB = mlir::dyn_cast<OpCall>(b);
        if (callA->opName != callB->opName)
          return mlir::failure();
        return unify_all(callA->arguments, callB->arguments, s);
      })
      .Case<Index>([b, &s](Index *idxA) {
        auto idxB = mlir::dyn_cast<Index>(b);
        if (idxA->signal != idxB->signal)
          return mlir::failure();
        return unify_all(idxA->indices, idxB->indices, s);
      })
      .Default([](auto) { return mlir::failure(); });
}

mlir::LogicalResult unify_all(llvm::ArrayRef<Symbol> as,
                              llvm::ArrayRef<Symbol> bs, Substitutions &s) {

  if (as.size() != bs.size())
    return mlir::failure();

  for (auto [a, b] : llvm::zip(as, bs)) {
    if (mlir::failed(unify(substitute(a, s), substitute(b, s), s))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

Symbol _single_subst(Symbol original, unsigned k, Symbol v) {
  return llvm::TypeSwitch<Symbol, Symbol>(original)
      .Case<Unknown>([k, v](Unknown *u) -> Symbol {
        if (u->n == k)
          return v;
        return u;
      })
      .Case<OpCall>([k, v](OpCall *c) -> Symbol {
        llvm::SmallVector<Symbol> substArgs;
        for (auto arg : c->arguments) {
          substArgs.push_back(_single_subst(arg, k, v));
        }
        return c->pool->func_call(c->opName, substArgs);
      })
      .Case<Index>([k, v](Index *i) -> Symbol {
        llvm::SmallVector<Symbol> indices;
        for (auto idx : i->indices) {
          indices.push_back(_single_subst(idx, k, v));
        }
        return i->pool->index(i->signal, indices);
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
  return original->pool->copy(orig);
}
} // namespace lleq
