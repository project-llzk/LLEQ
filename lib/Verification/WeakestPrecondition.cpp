/**
 * Copyright 2026 Project LLZK
 * SPDX-License-Identifier: Apache-2.0
 */

#include <llzk/Dialect/Bool/IR/Ops.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#define DEBUG_TYPE "weakest-precondition"

#include "Verification/TermUtils.h"
#include "Verification/VerificationUtils.h"
#include "Verification/WeakestPrecondition.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/TypeSwitch.h>

#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/Constrain/IR/Ops.h>
#include <llzk/Dialect/Felt/IR/Ops.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/SMT/IR/SMTOps.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Util/DynamicAPIntHelper.h>
#include <llzk/Util/TypeHelper.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>

#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <optional>
#include <vector>

using namespace llzk;
using namespace mlir;

using array::WriteArrayOp;
using component::MemberReadOp;
using component::MemberWriteOp;
using constrain::EmitEqualityOp;

namespace lleq {

struct LoopCounterInfo {
  cvc5::Term counter;
  Range range;
};

namespace {

// If `loop` is a perfectly nested loop, returns the innermost loop body and
// populates `loopInfo` with the bounds of each nest.
Block *nestedLoopBody(scf::ForOp loop, SmallVector<LoopCounterInfo> &loopInfo,
                      TermBuilder &builder) {
  loopInfo.push_back(LoopCounterInfo{
      builder.getExpression(loop.getInductionVar()),
      Range::fromValues(loop.getLowerBound(), loop.getUpperBound(),
                        loop.getStep(), builder)});
  if (auto first = dyn_cast<scf::ForOp>(loop.getBody()->front())) {
    return nestedLoopBody(first, loopInfo, builder);
  }
  return loop.getBody();
}

struct FailingCore {
  SmallVector<cvc5::Term> terms;
  SmallVector<Annotation> annotations;
};

cvc5::Term valueInRange(cvc5::Term value, Range range, cvc5::TermManager &mgr) {
  auto [lb, ub, step] = range;
  auto lBound = mgr.mkTerm(cvc5::Kind::LEQ, {lb, value});
  auto uBound = mgr.mkTerm(cvc5::Kind::LT, {value, ub});
  auto sBound =
      mgr.mkTerm(cvc5::Kind::EQUAL,
                 {mgr.mkTerm(cvc5::Kind::INTS_MODULUS,
                             {mgr.mkTerm(cvc5::Kind::SUB, {value, lb}), step}),
                  mgr.mkInteger(0)});
  return conjunctAll({lBound, uBound, sBound}, mgr);
}

// Returns the lexicographic prefix of the nested iteration space induced by
// `counters` and `ranges`.
//
// With `stepIter == false`, this is the set of tuples strictly before the
// current tuple `counters`.
//
// With `stepIter == true`, this is the set of tuples strictly before the
// lexicographic successor of `counters`, i.e. the visited region after the
// current body iteration executes once.
cvc5::Term getLoopAntecedent(ArrayRef<cvc5::Term> variables,
                             ArrayRef<cvc5::Term> counters,
                             ArrayRef<Range> ranges, cvc5::TermManager &mgr,
                             bool stepIter = false) {
  llzk::ensure(variables.size() == counters.size() &&
                   counters.size() == ranges.size(),
               "mismatched loop info");

  auto successorCounter = [&](int dim) -> cvc5::Term {
    if (!stepIter) {
      return counters[dim];
    }

    cvc5::Term result = counters[dim];
    for (int carryAt = counters.size() - 1; carryAt >= 0; carryAt--) {
      SmallVector<cvc5::Term> carryCondition;

      // All inner counters must be exhausted, otherwise the carry occurs at a
      // deeper nesting level.
      for (int inner = carryAt + 1; inner < counters.size(); inner++) {
        auto [innerLb, innerUb, innerStep] = ranges[inner];
        auto innerNext =
            mgr.mkTerm(cvc5::Kind::ADD, {counters[inner], innerStep});
        carryCondition.push_back(
            mgr.mkTerm(cvc5::Kind::NOT,
                       {mgr.mkTerm(cvc5::Kind::LT, {innerNext, innerUb})}));
      }

      // The carry lands here only if this counter can still advance.
      auto [carryLb, carryUb, carryStep] = ranges[carryAt];
      auto carryNext =
          mgr.mkTerm(cvc5::Kind::ADD, {counters[carryAt], carryStep});
      carryCondition.push_back(
          mgr.mkTerm(cvc5::Kind::LT, {carryNext, carryUb}));

      cvc5::Term successorValue = counters[dim];
      if (dim == carryAt) {
        successorValue =
            mgr.mkTerm(cvc5::Kind::ADD, {counters[dim], ranges[dim].step});
      } else if (dim > carryAt) {
        successorValue = ranges[dim].lb;
      }

      result = mgr.mkTerm(cvc5::Kind::ITE, {conjunctAll(carryCondition, mgr),
                                            successorValue, result});
    }

    return result;
  };

  SmallVector<cvc5::Term> disjuncts;
  for (auto [i, info] :
       llvm::enumerate(llvm::zip(variables, counters, ranges))) {
    auto [x, k, r] = info;
    SmallVector<cvc5::Term> conjuncts;
    conjuncts.push_back(
        valueInRange(x, Range{r.lb, successorCounter(i), r.step}, mgr));
    for (int j = 0; j < i; j++) {
      conjuncts.push_back(
          mgr.mkTerm(cvc5::Kind::EQUAL, {successorCounter(j), variables[j]}));
    }
    for (int j = i + 1; j < variables.size(); j++) {
      conjuncts.push_back(valueInRange(variables[j], ranges[j], mgr));
    }
    disjuncts.push_back(conjunctAll(conjuncts, mgr));
  }

  return disjunctAll(disjuncts, mgr);
}

cvc5::Term currentIterationTuple(ArrayRef<cvc5::Term> variables,
                                 ArrayRef<cvc5::Term> counters,
                                 cvc5::TermManager &mgr) {
  llzk::ensure(variables.size() == counters.size(), "mismatched loop info");

  SmallVector<cvc5::Term> equalities;
  for (auto [x, k] : llvm::zip(variables, counters)) {
    equalities.push_back(mgr.mkTerm(cvc5::Kind::EQUAL, {x, k}));
  }
  return conjunctAll(equalities, mgr);
}

// Turn `predicate(vars...)` into `forall xs, bound(xs...) => predicate(xs...)`
cvc5::Term
quantifyPredicate(cvc5::Term predicate, ArrayRef<cvc5::Term> vars,
                  std::function<cvc5::Term(ArrayRef<cvc5::Term>)> bound,
                  cvc5::TermManager &mgr) {
  std::vector<cvc5::Term> xs;
  for (int i = 0; i < vars.size(); i++) {
    xs.push_back(mgr.mkVar(mgr.getIntegerSort(), "x" + std::to_string(i)));
  }

  auto predicateBound = bound(xs);
  return mgr.mkTerm(
      cvc5::Kind::FORALL,
      {mgr.mkTerm(cvc5::Kind::VARIABLE_LIST, xs),
       mgr.mkTerm(cvc5::Kind::IMPLIES,
                  {predicateBound,
                   predicate.substitute({vars.begin(), vars.end()}, xs)})});
}

FailingCore getFailingCore(cvc5::Term invariant,
                           const ConjunctionTerm &postcondition,
                           TermBuilder &builder, llzk::Field field) {
  FailingCore core;
  auto &mgr = builder.manager();
  for (auto conjunction : postcondition.conjuncts) {
    auto antecedent = conjunctAll(conjunction.antecedents, mgr);
    for (auto [consequent, annotation] :
         llvm::zip(conjunction.consequents, conjunction.annotations)) {
      auto fails =
          mgr.mkTerm(cvc5::Kind::IMPLIES,
                     {conjunctAll({invariant, antecedent}, mgr), consequent});
      if (!checkUnsatWithPortfolio(buildSMTQuery(fails, builder, field))) {
        core.terms.push_back(consequent);
        llzk::ensure(annotation.has_value(),
                     "cannot produce unannotated failing core");
        core.annotations.push_back(*annotation);
      }
    }
  }
  return core;
}

} // namespace

FailureOr<cvc5::Term> WeakestPreconditionAnalysis::computeInvariant(
    scf::ForOp loop, const ConjunctionTerm &postcondition) {
  SmallVector<LoopCounterInfo> loopInfo;
  auto *body = nestedLoopBody(loop, loopInfo, builder);

  // Maps a witness array written in the loop to the full write index tuple.
  SmallVector<std::pair<component::MemberWriteOp, SmallVector<Value>>>
      witnessWrites;
  // DenseMap<component::MemberWriteOp, SmallVector<Value>> witnessWrites;
  body->walk([&witnessWrites](array::WriteArrayOp write) {
    auto dest = getArrayDestination(write.getArrRef());
    if (dest.has_value()) {
      witnessWrites.push_back(
          {*dest, SmallVector<Value>(write.getIndices().begin(),
                                     write.getIndices().end())});
    }
  });

  SmallVector<cvc5::Term> loopCounters;
  SmallVector<Range> loopBounds;
  for (auto [counter, bound] : loopInfo) {
    loopCounters.push_back(counter);
    loopBounds.push_back(bound);
  }

  SmallVector<cvc5::Term> predicates;
  SmallVector<SmallVector<cvc5::Term>> indices;
  SmallVector<ArrayShape> arrayShapes;
  for (auto [write, writeIndices] : witnessWrites) {
    auto memberDef = write.getMemberDefOp(tables)->get();
    auto arr_w_i =
        builder.arrayRead(builder.getConstant(memberDef, true), writeIndices);
    auto arr_c_i =
        builder.arrayRead(builder.getConstant(memberDef, false), writeIndices);
    predicates.push_back(builder.assertEqual(arr_w_i, arr_c_i));
    indices.push_back(llvm::map_to_vector(writeIndices, [this](Value index) {
      return builder.getExpression(index);
    }));

    auto arrType = dyn_cast<array::ArrayType>(memberDef.getType());
    llzk::ensure(static_cast<bool>(arrType),
                 "expected witness write destination to be an array");
    arrayShapes.push_back(
        ArrayShape(arrType.getShape().begin(), arrType.getShape().end()));
  }

  // Filter out which predicates are inductive
  SmallVector<cvc5::Term> inductivePredicates;
  // Conjuncts necessary to strengthen the inductive invariant
  // forall x in range(lb, i, step), forall sig, sig_c[x] == sig_w[x]
  cvc5::Term bodyPrecondition = quantifyPredicate(
      conjunctAll(predicates, mgr), loopCounters,
      [this, &loopCounters, &loopBounds](auto xs) {
        return getLoopAntecedent(xs, loopCounters, loopBounds, mgr);
      },
      mgr);

  SmallVector<cvc5::Term> strengthenings;
  for (auto [shape, predicate, indexTuple] :
       llvm::zip(arrayShapes, predicates, indices)) {
    // We can strengthen the invariant to say the array is equal outside the
    // slice visited by the for loop as well (note: this isn't quite right if,
    // e.g., the loop isn't a basic "step 1, write to arr[i]", but its pretty
    // hard to do much better in general)

    // Add a strengthening that asserts that the array outside the slice visited
    // by the loop is equivalent. So for an array write at index f(k...),
    // say something like: forall x..., (x \not\in R) /\ (f(x...) in array
    // bounds) -> predicate

    auto missesSlice = [this, &loopBounds, &indexTuple, &loopCounters,
                        &shape](ArrayRef<cvc5::Term> xs) {
      SmallVector<cvc5::Term> xNotInRange;
      for (auto [x, range] : llvm::zip(xs, loopBounds)) {
        xNotInRange.push_back(valueInRange(x, range, mgr).notTerm());
      }
      SmallVector<cvc5::Term> arrayBounds;
      for (auto [index, extent] : llvm::zip(indexTuple, shape)) {
        auto arrayAccessIndex =
            index.substitute({loopCounters.begin(), loopCounters.end()}, xs);
        arrayBounds.push_back(valueInRange(
            arrayAccessIndex,
            Range{mgr.mkInteger(0), mgr.mkInteger(extent), mgr.mkInteger(1)},
            mgr));
      }

      return mgr.mkTerm(cvc5::Kind::AND, {disjunctAll(xNotInRange, mgr),
                                          conjunctAll(arrayBounds, mgr)});
    };
    strengthenings.push_back(
        quantifyPredicate(predicate, loopCounters, missesSlice, mgr));

    // TODO: do something smarter here
    // strengthenings.push_back(
    //     quantifyPredicate(predicate, i, builder.getInteger(0), lb, step));
    // strengthenings.push_back(
    //     quantifyPredicate(predicate, i, ub, builder.getInteger(size), step));
  }

  auto strengthened = conjunctAll(strengthenings, mgr);
  auto strengthenedPrecondition =
      conjunctAll({bodyPrecondition, strengthened}, mgr);

  for (auto predicate : predicates) {
    auto postcondition = ConjunctionTerm::of(quantifyPredicate(
        predicate, loopCounters,
        [this, &loopCounters, &loopBounds](auto xs) {
          return disjunctAll(
              {getLoopAntecedent(xs, loopCounters, loopBounds, mgr),
               currentIterationTuple(xs, loopCounters, mgr)},
              mgr);
        },
        mgr));

    LLVM_DEBUG(llvm::dbgs() << postcondition.buildTerm(mgr).toString() << "\n");

    // Verify {strengthenedPrecondition} loopBody {postcondition} to show the
    // predicate is inductive
    calculateWP(body, postcondition);
    auto isInductive =
        mgr.mkTerm(cvc5::Kind::IMPLIES,
                   {strengthenedPrecondition, postcondition.buildTerm(mgr)});

    auto query = buildSMTQuery(isInductive, builder, field);
    LLVM_DEBUG({
      llvm::dbgs() << "Checking whether [" << predicate.toString()
                   << "] is inductive\n";
    });
    if (checkUnsatWithPortfolio(query)) {
      inductivePredicates.push_back(predicate);
    }
  }

  // Check that the resulting predicate entails the postcondition
  auto inductiveInvariant = conjunctAll(inductivePredicates, mgr);
  auto withinLoopBounds = [this, &loopCounters,
                           &loopBounds](ArrayRef<cvc5::Term> xs) {
    SmallVector<cvc5::Term> xInBound =
        llvm::map_to_vector(llvm::zip(xs, loopBounds), [&](auto pair) {
          auto [x, range] = pair;
          return valueInRange(x, range, mgr);
        });
    return conjunctAll(xInBound, mgr);
  };
  inductiveInvariant = quantifyPredicate(inductiveInvariant, loopCounters,
                                         withinLoopBounds, mgr);

  auto entailsPostcondition = postcondition;
  entailsPostcondition.addAntecedent(inductiveInvariant);
  entailsPostcondition.addAntecedent(strengthened);

  auto query =
      buildSMTQuery(entailsPostcondition.buildTerm(mgr), builder, field);

  if (checkUnsatWithPortfolio(query)) {
    return strengthened;
  } else {
    auto failingCore =
        getFailingCore(conjunctAll({inductiveInvariant, strengthened}, mgr),
                       postcondition, builder, field);
    for (auto [ann, elem] :
         llvm::zip(failingCore.annotations, failingCore.terms)) {
      llvm::dbgs() << "Loop invariant was not strong enough to prove: "
                   << elem.toString() << "\n";
      // NOTE: Just asserting the full thing, even when its an array, should be
      // fine. Asserting equivalence of the full array should never overlap with
      // something provable by the invariant, because the "slice" strengthenings
      // we added above already covered the "rest" of any array mentioned in the
      // invariant.
      strengthenings.push_back(elem);
    }
    // NOTE: The strengthened thing should trivially entail the postcondition
    // since we've added everything that was missing
    return conjunctAll(strengthenings, mgr);
  }

  // TODO: add strengthenings
}

void WeakestPreconditionAnalysis::addEquivalentMember(
    component::MemberDefOp memberDef) {
  builder.addEquivalentMember(memberDef);
}

static inline bool valueIsSignalRead(Value val, SymbolTableCollection &tables) {
  if (isa<BlockArgument>(val)) {
    return false;
  }
  if (auto memberRead = val.getDefiningOp<MemberReadOp>()) {
    auto memberDef = memberRead.getMemberDefOp(tables);
    if (failed(memberDef)) {
      return false;
    }
    return true;
    // return memberDef->get().getSignal();
  }
  return false;
}

static inline bool valueIsSignalWrite(Value val,
                                      SymbolTableCollection &tables) {
  for (auto use : val.getUsers()) {
    if (auto memberWrite = dyn_cast<MemberWriteOp>(use)) {
      auto memberDef = memberWrite.getMemberDefOp(tables);
      if (failed(memberDef)) {
        return false;
      }
      return true;
      // return memberDef->get().getSignal();
    }
  }
  return false;
}

// TODO: Use TermBuilder to populate expressions instead of substitution
void WeakestPreconditionAnalysis::calculateWP(Operation *op,
                                              ConjunctionTerm &postcondition) {
  llvm::TypeSwitch<mlir::Operation *, void>(op)
      .Case<component::CreateStructOp>(
          [&postcondition](auto) { return postcondition; })
      .Case<MemberWriteOp>([this, &postcondition](MemberWriteOp writeOp) {
        postcondition.addAntecedent(builder.assertEqual(
            builder.getConstant(writeOp.getMemberDefOp(tables)->get(),
                                /*isWitness=*/true),
            writeOp.getVal()));
      })
      .Case<WriteArrayOp>([this, &postcondition](WriteArrayOp writeOp) {
        auto arr = writeOp.getArrRef();
        SmallVector<Value> indices(writeOp.getIndices().begin(),
                                   writeOp.getIndices().end());
        auto value = writeOp.getRvalue();
        if (valueIsSignalRead(arr, tables) || valueIsSignalWrite(arr, tables)) {
          // TODO: Make this behavior configurable
          postcondition.addAntecedent(
              builder.assertEqual(builder.arrayRead(arr, indices), value));
          return;
        }
        postcondition.substitute(builder.getConstant(arr),
                                 builder.arrayWrite(arr, indices, value));
      })
      .Case<constrain::EmitEqualityOp>(
          [this, &postcondition](EmitEqualityOp eqOp) {
            postcondition.addAntecedent(
                builder.assertEqual(eqOp.getLhs(), eqOp.getRhs()));
          })
      .Case<scf::IfOp>([this, &postcondition](scf::IfOp op) {
        calculateWP(op, postcondition);
      })
      .Case<UnrealizedConversionCastOp>([this](auto) { return; })
      .Case<scf::ForOp>([this, &postcondition](scf::ForOp op) {
        auto invariant = computeInvariant(op, postcondition);
        llzk::ensure(succeeded(invariant),
                     "failed to infer invariant for loop");
        // It should already be the case that invariant => postcondition
        postcondition = ConjunctionTerm::of(*invariant);
      })
      .Case<boolean::AssertOp>([this, &postcondition](boolean::AssertOp op) {
        postcondition.addAntecedent(builder.getExpression(op.getCondition()));
      })
      .Case<smt::AssertOp>([this, &postcondition](smt::AssertOp op) {
        postcondition.addAntecedent(builder.getExpression(op.getInput()));
      })
      .Case<llzk::function::CallOp>([this, &postcondition](
                                        llzk::function::CallOp call) {
        if (call.calleeIsConstrain()) {
          // @constrain(%subcmp, %args...) => (assert (= %subcmp
          // (init-"subcmp" %args...)))
          auto target = call.getCalleeTarget(tables);
          llzk::ensure(succeeded(target), "failed to resolve callee target");
          auto subcmpVal = call.getArgOperands().front();
          auto subcmp =
              target->get()->getParentOfType<component::StructDefOp>();
          SmallVector<Value> args = call.getArgOperands().drop_front();
          postcondition.addAntecedent(
              builder.assertEqual(subcmpVal, builder.initSubcmp(subcmp, args)));
        } else {
          // Do the default (there's gotta be a better way)
          llzk::ensure(call.calleeIsCompute(),
                       "arbitrary function calls not supported");
          auto expression = builder.getExpression(call.getResult(0));
          postcondition.substitute(builder.getConstant(call->getResult(0)),
                                   expression);
        }
      })
      .Default([this, &postcondition](auto op) {
        auto expression = builder.getExpression(op->getResult(0));
        postcondition.substitute(builder.getConstant(op->getResult(0)),
                                 expression);
      });
}

void WeakestPreconditionAnalysis::calculateWP(Block *block,
                                              ConjunctionTerm &postcondition) {
  // TODO: also return yielded values
  for (auto &op : llvm::iterator_range(block->rbegin(), block->rend())) {
    if (&op == block->getTerminator()) {
      continue;
    }
    calculateWP(&op, postcondition);
  }
}

void WeakestPreconditionAnalysis::calculateWP(mlir::scf::IfOp ifOp,
                                              ConjunctionTerm &postcondition) {
  auto condition = builder.getConstant(ifOp.getCondition());
  auto notCondition = mgr.mkTerm(cvc5::Kind::NOT, {condition});

  ConjunctionTerm thenBranch{postcondition}, elseBranch{postcondition};
  calculateWP(&ifOp.getThenRegion().front(), thenBranch);
  calculateWP(&ifOp.getElseRegion().front(), elseBranch);

  thenBranch.addAntecedent(condition);
  elseBranch.addAntecedent(notCondition);

  thenBranch.addConjuncts(elseBranch);
  postcondition = thenBranch;
}

SmallVector<cvc5::Term> getArrayExtents(array::ArrayType type,
                                        TermBuilder &builder) {
  auto sizes = type.getDimensionSizes();
  auto shape = type.getShape();
  SmallVector<cvc5::Term> extents;
  for (int i = 0; i < sizes.size(); i++) {
    if (type.isDynamicDim(i)) {
      if (auto symbolRef = dyn_cast<SymbolRefAttr>(sizes[i])) {
        extents.push_back(
            builder.getConstant(symbolRef.getLeafReference().strref()));
      }
    } else {
      extents.push_back(builder.getInteger(shape[i]));
    }
  }
  return extents;
}

ImplicationTerm WeakestPreconditionAnalysis::getPostcondition() {

  auto members = structDef.getMemberDefs();
  llzk::ensure(!members.empty(),
               "cannot build postcondition for struct with empty members");

  SmallVector<cvc5::Term> memberEquivs;
  SmallVector<std::optional<Annotation>> annotations;
  for (auto memberDef : members) {
    auto witnessSym = builder.getConstant(memberDef, true);
    auto constraintSym = builder.getConstant(memberDef, false);
    if (auto arrType = dyn_cast<llzk::array::ArrayType>(memberDef.getType())) {
      SmallVector<Range> slices;
      auto extents = getArrayExtents(arrType, builder);
      for (auto extent : extents) {
        slices.push_back(
            {builder.getInteger(0), extent, builder.getInteger(1)});
      }
      annotations.push_back(Annotation{true, std::move(slices)});
    } else {
      annotations.push_back(Annotation{false, std::nullopt});
    }
    memberEquivs.push_back(builder.assertEqual(witnessSym, constraintSym));
  }

  return ImplicationTerm{{}, memberEquivs, annotations};
}

void WeakestPreconditionAnalysis::populateVerificationConditions() {
  llzk::ensure(succeeded(ensureProductFunc(
                   structDef->getParentOfType<ModuleOp>(), structDef)),
               "failed to align product func");

  auto postcondition = ConjunctionTerm::of(getPostcondition());
  calculateWP(&structDef.getProductFuncOp().getFunctionBody().front(),
              postcondition);

  verificationConditions = postcondition.buildTerm(mgr);
  extraDecls = builder.getExtraDecls(verificationConditions);
  declBounds = builder.getDeclBounds(extraDecls, field.prime());
}

cvc5::Term WeakestPreconditionAnalysis::generateVerificationConditions() {
  llzk::ensure(succeeded(ensureProductFunc(
                   structDef->getParentOfType<ModuleOp>(), structDef)),
               "failed to align product func");
  auto postcondition = ConjunctionTerm::of(getPostcondition());
  calculateWP(&structDef.getProductFuncOp().getFunctionBody().front(),
              postcondition);

  return postcondition.buildTerm(mgr);
}

void WeakestPreconditionAnalysis::emit(llvm::raw_ostream &os) {
  auto verificationConditions = generateVerificationConditions();
  auto extraDecls = builder.getExtraDecls(verificationConditions);
  auto bounds = builder.getDeclBounds(extraDecls, field.prime());

  os << "(set-logic ALL)\n";
  builder.emitSubcmpDeclarations(os);

  os << "; Extra declarations\n";
  for (auto decl : extraDecls) {
    os << "(declare-const " << decl.toString() << " "
       << decl.getSort().toString() << ")\n";
  }

  os << "; Extra bounds\n";
  for (auto bound : bounds) {
    os << "(assert " << bound.toString() << ")\n";
  }

  os << "; Verification condition\n";
  os << "(assert " << verificationConditions.notTerm().toString() << ")\n";
  os << "(check-sat)\n";
}

} // namespace lleq
