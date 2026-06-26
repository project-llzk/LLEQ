/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/WeakestPrecondition.h"
#include "Analysis/ScalarSymbolAnalysis.h"
#include "Verification/TermUtils.h"
#include "Verification/VerificationUtils.h"

#include <fstream>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/TypeSwitch.h>

#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/Program.h>
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
#include <unistd.h>
#include <vector>

using namespace llzk;
using namespace mlir;

using array::ReadArrayOp;
using array::WriteArrayOp;
using component::MemberReadOp;
using component::MemberWriteOp;
using constrain::EmitEqualityOp;
using felt::FeltConstantOp;

namespace lleq {

struct LoopCounterInfo {
  cvc5::Term counter;
  Range range;
};

namespace {

std::string resolveZ3Path() {
  if (std::optional<std::string> envPath =
          llvm::sys::Process::GetEnv("LLEQ_Z3")) {
    if (llvm::sys::fs::can_execute(*envPath)) {
      return *envPath;
    }
    llvm::report_fatal_error(StringRef{"LLEQ_Z3 is set to '"} + *envPath +
                             "', but that path is not executable");
  }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  auto solverPath = llvm::sys::findProgramByName("z3");
#pragma clang diagnostic pop
  if (!solverPath) {
    llvm::report_fatal_error(
        "could not find an executable z3 binary; install z3 or set LLEQ_Z3 "
        "to the full path of the solver binary");
  }
  return *solverPath;
}

std::string buildSMTQuery(cvc5::Term query, TermBuilder &builder,
                          llzk::Field field) {
  auto extraDecls = builder.getExtraDecls(query);
  auto bounds = builder.getDeclBounds(extraDecls, field.prime());
  std::string queryStr;
  llvm::raw_string_ostream os{queryStr};

  os << "(set-logic ALL)\n";
  builder.emitSubcmpDeclarations(os);
  for (auto decl : extraDecls) {
    os << "(declare-const " << decl.toString() << " "
       << decl.getSort().toString() << ")\n";
  }
  for (auto bound : bounds) {
    os << "(assert " << bound.toString() << ")\n";
  }
  os << "(assert " << query.notTerm().toString() << ")\n";
  os << "(check-sat)\n";
  return queryStr;
}

bool checkUnsatWithZ3(StringRef query) {
  int stdinFd = -1;
  SmallString<128> tempStdin;
  llzk::ensure(llvm::sys::fs::createTemporaryFile("lleq-z3-stdin", "smt2",
                                                  stdinFd, tempStdin) ==
                   std::error_code{},
               "failed to create temporary file for z3 stdin");
  int stdoutFd = -1;
  SmallString<128> tempStdout;
  llzk::ensure(llvm::sys::fs::createTemporaryFile("lleq-z3-stdout", "txt",
                                                  stdoutFd, tempStdout) ==
                   std::error_code{},
               "failed to create temporary file for z3 stdout");

  std::ofstream os{tempStdin.c_str()};
  os.write(query.data(), query.size());
  os.close();
  if (stdinFd >= 0) {
    ::close(stdinFd);
  }
  if (stdoutFd >= 0) {
    ::close(stdoutFd);
  }

  auto solverPath = resolveZ3Path();
  SmallVector<StringRef> args{solverPath, "-smt2", tempStdin};

  std::string error;
  auto code = llvm::sys::ExecuteAndWait(solverPath, args,
                                        /*Env=*/std::nullopt,
                                        /*Redirects=*/
                                        {"", std::string{tempStdout}, ""}, 0, 0,
                                        &error);
  if (code) {
    llvm::report_fatal_error(error.c_str());
  }

  std::ifstream is{tempStdout.c_str()};
  std::string result;
  is >> result;
  llvm::sys::fs::remove(tempStdin);
  llvm::sys::fs::remove(tempStdout);
  return result == "unsat";
}

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

// Turn `predicate(var)` into `forall x in range(lb, ub, step), predicate(x)`
// cvc5::Term quantifyPredicate(cvc5::Term predicate,
//                              SmallVector<cvc5::Term> counters,
//                              SmallVector<Range> ranges, cvc5::TermManager
//                              &mgr, bool stepIter = false) {
//   llzk::ensure(counters.size() == ranges.size(), "mismatched loop info");

//   SmallVector<cvc5::Term> vars;
//   for (int i = 0; i < counters.size(); i++) {
//     vars.push_back(mgr.mkVar(mgr.getIntegerSort(), "x" + std::to_string(i)));
//   }

//   auto bound = getLoopAntecedent(vars, counters, ranges, mgr, stepIter);
//   return mgr.mkTerm(
//       cvc5::Kind::FORALL,
//       {mgr.mkTerm(cvc5::Kind::VARIABLE_LIST, {vars.begin(), vars.end()}),
//        mgr.mkTerm(
//            cvc5::Kind::IMPLIES,
//            {bound, predicate.substitute({counters.begin(), counters.end()},
//                                         {vars.begin(), vars.end()})})});
// }

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
      if (!checkUnsatWithZ3(buildSMTQuery(fails, builder, field))) {
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
  DenseMap<component::MemberWriteOp, SmallVector<Value>> witnessWrites;
  body->walk([&witnessWrites](array::WriteArrayOp write) {
    auto dest = getArrayDestination(write.getArrRef());
    if (dest.has_value()) {
      witnessWrites[*dest] = SmallVector<Value>(write.getIndices().begin(),
                                                write.getIndices().end());
    }
  });

  // llzk::ensure(loopInfo.size() == 1, "nested loop support in-progress");

  SmallVector<cvc5::Term> loopCounters;
  SmallVector<Range> loopBounds;
  for (auto [counter, bound] : loopInfo) {
    loopCounters.push_back(counter);
    loopBounds.push_back(bound);
  }

  // auto [i, range] = loopInfo.front();
  // auto [lb, ub, step] = range;

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

    llvm::dbgs() << postcondition.buildTerm(mgr).toString() << "\n";

    // Verify {strengthenedPrecondition} loopBody {postcondition} to show the
    // predicate is inductive
    calculateWP(body, postcondition);
    auto isInductive =
        mgr.mkTerm(cvc5::Kind::IMPLIES,
                   {strengthenedPrecondition, postcondition.buildTerm(mgr)});
    // llvm::dbgs() << "Is inductive: " << isInductive.toString() << "\n";
    auto query = buildSMTQuery(isInductive, builder, field);
    // llvm::dbgs() << invariant.toString() << "\n";
    llvm::dbgs() << "Checking whether [" << predicate.toString()
                 << "] is inductive\n";
    llvm::dbgs() << query << "\n";
    if (checkUnsatWithZ3(query)) {
      llvm::dbgs() << "Predicate " << predicate.toString() << " is inductive\n";
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

  llvm::dbgs() << "Checking invariant " << inductiveInvariant.toString()
               << "\n";

  auto entailsPostcondition = postcondition;
  entailsPostcondition.addAntecedent(inductiveInvariant);
  entailsPostcondition.addAntecedent(strengthened);

  auto query =
      buildSMTQuery(entailsPostcondition.buildTerm(mgr), builder, field);
  llvm::dbgs() << query << "\n";
  if (checkUnsatWithZ3(query)) {
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
    // TODO: Technically we should check here that the strengthened thing
    // entails the postcondition; though it trivially should since we've added
    // everything that was missing
    return conjunctAll(strengthenings, mgr);
  }

  // TODO: add strengthenings
}

void WeakestPreconditionAnalysis::addEquivalentMember(
    component::MemberDefOp memberDef) {
  builder.addEquivalentMember(memberDef);
}

// TODO: I *think* its enough to implement subcmp calls to @compute/@constrain
// in here since writing to the subcmp member should handle the assertion, and:
// (1) If the top struct was aligned mechanically the SSA values being written
// to _w and _c should be distinct, and
// (2) Otherwise if they aren't distinct, it should still be correct to assert
// these two (init-...) invocations equal?
mlir::FailureOr<cvc5::Term>
WeakestPreconditionAnalysis::getExpression(Operation *op) {
  static llvm::DenseMap<StringRef, cvc5::Kind> opToTermKind = {
      {"felt.add", cvc5::Kind::ADD},
      {"felt.sub", cvc5::Kind::SUB},
      {"felt.mul", cvc5::Kind::MULT},
      {"felt.smod", cvc5::Kind::INTS_MODULUS},
      {"felt.sintdiv", cvc5::Kind::INTS_DIVISION},
      {"felt.div", cvc5::Kind::INTS_DIVISION}};

  if (auto it = opToTermKind.find(op->getName().getStringRef());
      it != opToTermKind.end()) {
    SmallVector<cvc5::Term> operandTerms{
        llvm::map_to_vector(op->getOperands(), [this](Value value) {
          return getExpression(value);
        })};
    return mgr.mkTerm(it->second, {operandTerms.begin(), operandTerms.end()});
  }

  return llvm::TypeSwitch<Operation *, FailureOr<cvc5::Term>>(op)
      .Case<MemberReadOp>([this](MemberReadOp read) {
        return builder.getConstant(read.getMemberDefOp(tables)->get(),
                                   isWitnessOp(read));
      })
      .Case<ReadArrayOp>([this](ReadArrayOp read) {
        SmallVector<cvc5::Term> indices =
            llvm::map_to_vector(read.getIndices(), [this](Value index) {
              return builder.getExpression(index);
            });
        return builder.arrayRead(builder.getExpression(read.getArrRef()),
                                 indices);
      })
      .Case<FeltConstantOp>([this](FeltConstantOp constOp) {
        SmallString<64> str;
        constOp.getValue().getValue().toStringUnsigned(str);
        return mgr.mkInteger(std::string{str});
      })
      .Case<arith::ConstantIntOp, arith::ConstantIndexOp>([this](auto constOp) {
        auto val = dyn_cast<IntegerAttr>(constOp.getValue()).getValue();
        return builder.getInteger(val);
      })
      .Case<array::CreateArrayOp>([this](array::CreateArrayOp createArr) {
        // If the array is written to exactly one struct member later, just
        // materialize a symbol for that directly
        auto destination = getArrayDestination(createArr.getResult());
        if (destination.has_value()) {
          return builder.getConstant(destination->getMemberName(),
                                     createArr.getType(), true);
        }
        return builder.getConstant(createArr.getResult());
      })
      .Case<llzk::function::CallOp>([this](llzk::function::CallOp call) {
        // For now just deal with calls to @compute and error out on other
        // function calls
        llzk::ensure(call.calleeIsCompute(),
                     "arbitrary function calls not supported yet");
        auto target = call.getCalleeTarget(tables);
        llzk::ensure(succeeded(target), "failed to resolve callee target");
        SmallVector<Value> args = call.getArgOperands();
        return builder.initSubcmp(
            target->get()->getParentOfType<component::StructDefOp>(), args);
      })
      .Default([op](auto) {
        return failure();
        // llvm::report_fatal_error("unknown op: " +
        // op->getName().getStringRef());
      });
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
          auto expression = getExpression(call);
          llzk::ensure(succeeded(expression),
                       "unknown op: " + call->getName().getStringRef());
          postcondition.substitute(builder.getConstant(call->getResult(0)),
                                   *expression);
        }
      })
      .Default([this, &postcondition](auto op) {
        auto expression = builder.getExpression(op->getResult(0));
        // llzk::ensure(succeeded(expression),
        //              "unknown op: " + op->getName().getStringRef());
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
      for (int64_t extent : arrType.getShape()) {
        slices.push_back({builder.getInteger(0), builder.getInteger(extent),
                          builder.getInteger(1)});
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

  llvm::dbgs() << structDef << "\n----\n";

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
  os << "(check-sat)\n(get-model)\n";
}

} // namespace lleq
