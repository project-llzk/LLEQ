/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/WeakestPrecondition.h"
#include "Analysis/ScalarSymbolAnalysis.h"
#include "Transforms/LLEQWhileToFor.h"
#include "Verification/TermUtils.h"
#include "Verification/VerificationUtils.h"

#include <fstream>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/TypeSwitch.h>

#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FileSystem.h>
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
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>

#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <optional>
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

using transform::ForOpInfo;

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
  SmallString<128> tempStdin;
  SmallString<128> tempStdout;
  int tempStdinFd;
  if (std::error_code ec = llvm::sys::fs::createTemporaryFile(
          "lleq-z3-query", "smt2", tempStdinFd, tempStdin)) {
    llvm::report_fatal_error(llvm::Twine(
        "failed to create temporary Z3 query file: " + ec.message()));
  }
  if (std::error_code ec = llvm::sys::fs::createTemporaryFile(
          "lleq-z3-output", "txt", tempStdout)) {
    llvm::sys::fs::remove(tempStdin);
    llvm::report_fatal_error(llvm::Twine(
        "failed to create temporary Z3 output file: " + ec.message()));
  }

  llvm::raw_fd_ostream os{tempStdinFd, /*shouldClose=*/true};
  os << query.data();
  os.close();

  auto solverPath = resolveZ3Path();
  SmallVector<StringRef> args{solverPath, "-in", "-smt2"};
  std::string tempStdinStr = tempStdin.str().str();
  std::string tempStdoutStr = tempStdout.str().str();

  std::string error;
  auto code = llvm::sys::ExecuteAndWait(solverPath, args,
                                        /*Env=*/std::nullopt,
                                        /*Redirects=*/
                                        {tempStdinStr, tempStdoutStr, ""}, 0, 0,
                                        &error);
  if (code) {
    llvm::sys::fs::remove(tempStdin);
    llvm::sys::fs::remove(tempStdout);
    llvm::report_fatal_error(error.c_str());
  }

  std::ifstream is{tempStdoutStr};
  std::string result;
  is >> result;
  llvm::sys::fs::remove(tempStdin);
  llvm::sys::fs::remove(tempStdout);
  return result == "unsat";
}

Block *nestedLoopBody(scf::ForOp loop, SmallVector<ForOpInfo> &loopInfo) {
  loopInfo.push_back({loop.getLowerBound(), loop.getUpperBound(),
                      loop.getStep(), loop.getInductionVar(), 0});
  if (auto first = dyn_cast<scf::ForOp>(loop.getBody()->front())) {
    return nestedLoopBody(first, loopInfo);
  }
  return loop.getBody();
}

enum class LoopGuardKind { BeforeCurrent, AfterCurrent, AfterExit };

struct LoopCounterTerms {
  cvc5::Term counter;
  cvc5::Term lowerBound;
  cvc5::Term upperBound;
  cvc5::Term step;
};

struct FlatLoopIndex {
  cvc5::Term currentIndex;
  int64_t totalIterations;
};

std::optional<int64_t> getConstantIndex(Value value) {
  auto constOp = value.getDefiningOp<arith::ConstantIndexOp>();
  if (!constOp) {
    return std::nullopt;
  }
  return constOp.value();
}

cvc5::Term makeAnd(cvc5::TermManager &mgr, llvm::ArrayRef<cvc5::Term> terms) {
  if (terms.empty()) {
    return mgr.mkBoolean(true);
  }
  if (terms.size() == 1) {
    return terms.front();
  }
  return mgr.mkTerm(cvc5::Kind::AND, {terms.begin(), terms.end()});
}

cvc5::Term makeOr(cvc5::TermManager &mgr, llvm::ArrayRef<cvc5::Term> terms) {
  if (terms.empty()) {
    return mgr.mkBoolean(false);
  }
  if (terms.size() == 1) {
    return terms.front();
  }
  return mgr.mkTerm(cvc5::Kind::OR, {terms.begin(), terms.end()});
}

cvc5::Term makeStepAligned(cvc5::TermManager &mgr, TermBuilder &builder,
                           cvc5::Term value, const LoopCounterTerms &counter) {
  auto offset = mgr.mkTerm(cvc5::Kind::SUB, {value, counter.lowerBound});
  auto modulus = mgr.mkTerm(cvc5::Kind::INTS_MODULUS, {offset, counter.step});
  return mgr.mkTerm(cvc5::Kind::EQUAL, {modulus, builder.getInteger(0)});
}

cvc5::Term makeFullRange(cvc5::TermManager &mgr, TermBuilder &builder,
                         cvc5::Term value, const LoopCounterTerms &counter) {
  return makeAnd(mgr,
                 {
                     mgr.mkTerm(cvc5::Kind::LEQ, {counter.lowerBound, value}),
                     mgr.mkTerm(cvc5::Kind::LT, {value, counter.upperBound}),
                     makeStepAligned(mgr, builder, value, counter),
                 });
}

cvc5::Term makeActiveLoopGuard(cvc5::TermManager &mgr,
                               llvm::ArrayRef<LoopCounterTerms> counters) {
  SmallVector<cvc5::Term> bounds;
  for (const auto &counter : counters) {
    bounds.push_back(
        mgr.mkTerm(cvc5::Kind::LEQ, {counter.lowerBound, counter.counter}));
    bounds.push_back(
        mgr.mkTerm(cvc5::Kind::LT, {counter.counter, counter.upperBound}));
  }
  return makeAnd(mgr, bounds);
}

cvc5::Term makeNestedLoopGuard(cvc5::TermManager &mgr, TermBuilder &builder,
                               llvm::ArrayRef<LoopCounterTerms> counters,
                               llvm::ArrayRef<cvc5::Term> quantVars,
                               LoopGuardKind kind) {
  if (kind == LoopGuardKind::AfterExit) {
    SmallVector<cvc5::Term> ranges;
    for (auto [counter, var] : llvm::zip_equal(counters, quantVars)) {
      ranges.push_back(makeFullRange(mgr, builder, var, counter));
    }
    return makeAnd(mgr, ranges);
  }

  SmallVector<cvc5::Term> guardCases;
  for (size_t i = 0; i < counters.size(); ++i) {
    SmallVector<cvc5::Term> conjuncts;

    for (size_t j = 0; j < i; ++j) {
      conjuncts.push_back(
          mgr.mkTerm(cvc5::Kind::EQUAL, {quantVars[j], counters[j].counter}));
    }

    auto currentBound = counters[i].counter;
    if (kind == LoopGuardKind::AfterCurrent && i == counters.size() - 1) {
      currentBound =
          mgr.mkTerm(cvc5::Kind::ADD, {currentBound, counters[i].step});
    }

    conjuncts.push_back(
        mgr.mkTerm(cvc5::Kind::LEQ, {counters[i].lowerBound, quantVars[i]}));
    conjuncts.push_back(
        mgr.mkTerm(cvc5::Kind::LT, {quantVars[i], currentBound}));
    conjuncts.push_back(
        makeStepAligned(mgr, builder, quantVars[i], counters[i]));

    for (size_t j = i + 1; j < counters.size(); ++j) {
      conjuncts.push_back(
          makeFullRange(mgr, builder, quantVars[j], counters[j]));
    }

    guardCases.push_back(makeAnd(mgr, conjuncts));
  }
  return makeOr(mgr, guardCases);
}

std::optional<FlatLoopIndex>
detectRowMajorIndex(cvc5::TermManager &mgr, TermBuilder &builder,
                    llvm::ArrayRef<ForOpInfo> loopInfo, cvc5::Term index) {
  int64_t totalIterations = 1;
  SmallVector<int64_t> tripCounts;
  for (const auto &info : loopInfo) {
    auto lb = getConstantIndex(*info.lb);
    auto ub = getConstantIndex(*info.ub);
    auto step = getConstantIndex(*info.step);
    if (!lb || !ub || !step || *lb != 0 || *step != 1 || *ub <= 0) {
      return std::nullopt;
    }
    tripCounts.push_back(*ub);
    totalIterations *= *ub;
  }

  std::optional<cvc5::Term> expected;
  for (auto [idx, info] : llvm::enumerate(loopInfo)) {
    int64_t stride = 1;
    for (int64_t innerTripCount :
         llvm::ArrayRef(tripCounts).drop_front(idx + 1)) {
      stride *= innerTripCount;
    }

    auto counter = builder.getConstant(*info.ivar);
    auto term = stride == 1 ? counter
                            : mgr.mkTerm(cvc5::Kind::MULT,
                                         {counter, builder.getInteger(stride)});
    expected = expected ? mgr.mkTerm(cvc5::Kind::ADD, {*expected, term}) : term;
  }

  if (!expected || !(*expected == index)) {
    return std::nullopt;
  }
  return FlatLoopIndex{*expected, totalIterations};
}

} // namespace

TermBuilder::TermSet
WeakestPreconditionAnalysis::conjecturePredicates(mlir::scf::ForOp loop) {
  SmallVector<ForOpInfo> loopInfo;
  auto *body = nestedLoopBody(loop, loopInfo);

  // Maps a witness *array* that's written in the loop to the index at which its
  // written
  DenseMap<component::MemberWriteOp, Value> witnessWrites;
  body->walk([&witnessWrites](array::WriteArrayOp write) {
    llzk::ensure(write.getIndices().size() == 1,
                 "multidimensional arrays not yet supported");
    auto dest = getArrayDestination(write.getArrRef());
    if (dest.has_value()) {
      witnessWrites.insert({*dest, write.getIndices().front()});
    }
  });

  TermBuilder::TermSet invariants;
  if (witnessWrites.empty()) {
    return invariants;
  }

  std::vector<cvc5::Term> counters;
  std::vector<cvc5::Term> quantVars;
  SmallVector<LoopCounterTerms> counterTerms;
  for (auto [idx, info] : llvm::enumerate(loopInfo)) {
    auto counter = builder.getConstant(*info.ivar);
    auto lowerBound = builder.getExpression(*info.lb);
    auto upperBound = builder.getExpression(*info.ub);
    auto step = builder.getExpression(*info.step);

    counters.push_back(counter);
    quantVars.push_back(
        mgr.mkVar(mgr.getIntegerSort(), "x" + std::to_string(idx)));
    counterTerms.push_back({counter, lowerBound, upperBound, step});
  }

  auto substituteCounters = [&counters](cvc5::Term term,
                                        const std::vector<cvc5::Term> &terms) {
    return term.substitute(counters, terms);
  };

  auto makePredicateAtIndex = [this](component::MemberWriteOp write,
                                     cvc5::Term indexedTerm) {
    auto member = write.getMemberDefOp(tables)->get();
    auto arr_w_i =
        builder.arrayRead(builder.getConstant(member, true), indexedTerm);
    auto arr_c_i =
        builder.arrayRead(builder.getConstant(member, false), indexedTerm);
    return builder.assertEqual(arr_w_i, arr_c_i);
  };

  auto makePredicate = [this, &substituteCounters, &makePredicateAtIndex](
                           component::MemberWriteOp write, Value index,
                           const std::vector<cvc5::Term> &indexTerms) {
    auto indexedTerm =
        substituteCounters(builder.getExpression(index), indexTerms);
    return makePredicateAtIndex(write, indexedTerm);
  };

  std::vector<cvc5::Term> predicates;
  for (auto [write, index] : witnessWrites) {
    predicates.push_back(makePredicate(write, index, counters));
  }

  // all signals are equal at i
  auto allEqualUpToCurrent = predicates.size() == 1
                                 ? predicates.front()
                                 : mgr.mkTerm(cvc5::Kind::AND, predicates);

  auto quantifyPredicate = [this, &counterTerms, &quantVars,
                            &substituteCounters](cvc5::Term predicate,
                                                 LoopGuardKind kind) {
    auto guard =
        makeNestedLoopGuard(mgr, builder, counterTerms, quantVars, kind);
    auto body = substituteCounters(predicate, quantVars);
    return mgr.mkTerm(cvc5::Kind::FORALL,
                      {mgr.mkTerm(cvc5::Kind::VARIABLE_LIST, quantVars),
                       mgr.mkTerm(cvc5::Kind::IMPLIES, {guard, body})});
  };

  // Turn "all signals equal at current counters" into "all signals equal at
  // every nested counter tuple that has already executed".
  allEqualUpToCurrent =
      quantifyPredicate(allEqualUpToCurrent, LoopGuardKind::BeforeCurrent);

  // Now, filter out the ones that aren't loop invariants
  for (auto invariant : predicates) {
    auto invariantHeld = makeAnd(
        mgr, {makeActiveLoopGuard(mgr, counterTerms), allEqualUpToCurrent});

    // Calculate wp(B, I)
    auto precondition = ConjunctionTerm::of(
        quantifyPredicate(invariant, LoopGuardKind::AfterCurrent));
    calculateWP(body, precondition);

    // Check C /\ AllI => wp(B, I)
    auto isInvariant = mgr.mkTerm(cvc5::Kind::IMPLIES,
                                  {invariantHeld, precondition.buildTerm(mgr)});

    llvm::dbgs() << isInvariant.toString() << "\n";

    auto query = buildSMTQuery(isInvariant, builder, field);
    if (checkUnsatWithZ3(query)) {
      // Assume the invariant holds after exiting the loop
      invariants.insert(quantifyPredicate(invariant, LoopGuardKind::AfterExit));
    } else {
      return invariants;
    }
  }

  struct FlatPredicate {
    component::MemberWriteOp write;
    cvc5::Term currentIndex;
    int64_t totalIterations;
  };
  SmallVector<FlatPredicate> flatPredicates;
  if (loopInfo.size() > 1) {
    for (auto [write, index] : witnessWrites) {
      auto flatIndex = detectRowMajorIndex(mgr, builder, loopInfo,
                                           builder.getExpression(index));
      if (flatIndex) {
        flatPredicates.push_back(
            {write, flatIndex->currentIndex, flatIndex->totalIterations});
      }
    }
  }

  if (!flatPredicates.empty()) {
    auto flatVar = mgr.mkVar(mgr.getIntegerSort(), "idx");
    auto currentFlatIndex = flatPredicates.front().currentIndex;
    auto totalIterations = flatPredicates.front().totalIterations;

    auto makeFlatGuard = [this, &flatVar](cvc5::Term upperBound) {
      return makeAnd(
          mgr, {mgr.mkTerm(cvc5::Kind::LEQ, {builder.getInteger(0), flatVar}),
                mgr.mkTerm(cvc5::Kind::LT, {flatVar, upperBound})});
    };

    auto makeFlatPredicate = [this, &flatVar, &flatPredicates,
                              &makePredicateAtIndex]() {
      std::vector<cvc5::Term> conjuncts;
      for (auto flatPredicate : flatPredicates) {
        conjuncts.push_back(makePredicateAtIndex(flatPredicate.write, flatVar));
      }
      return conjuncts.size() == 1 ? conjuncts.front()
                                   : mgr.mkTerm(cvc5::Kind::AND, conjuncts);
    };

    auto quantifyFlatPredicate = [this, &flatVar, &makeFlatGuard,
                                  &makeFlatPredicate](cvc5::Term upperBound) {
      auto guard = makeFlatGuard(upperBound);
      return mgr.mkTerm(
          cvc5::Kind::FORALL,
          {mgr.mkTerm(cvc5::Kind::VARIABLE_LIST, {flatVar}),
           mgr.mkTerm(cvc5::Kind::IMPLIES, {guard, makeFlatPredicate()})});
    };

    auto flatAllEqualUpToCurrent = quantifyFlatPredicate(currentFlatIndex);
    auto invariantHeld = makeAnd(
        mgr, {makeActiveLoopGuard(mgr, counterTerms), flatAllEqualUpToCurrent});

    auto afterCurrent =
        mgr.mkTerm(cvc5::Kind::ADD, {currentFlatIndex, builder.getInteger(1)});
    auto precondition =
        ConjunctionTerm::of(quantifyFlatPredicate(afterCurrent));
    calculateWP(body, precondition);

    auto isInvariant = mgr.mkTerm(cvc5::Kind::IMPLIES,
                                  {invariantHeld, precondition.buildTerm(mgr)});
    auto query = buildSMTQuery(isInvariant, builder, field);
    if (checkUnsatWithZ3(query)) {
      invariants.insert(
          quantifyFlatPredicate(builder.getInteger(totalIterations)));
    }
  }

  return invariants;
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
        llzk::ensure(read.getIndices().size() == 1,
                     "multidimensional arrays are not supported");
        return builder.arrayRead(
            builder.getExpression(read.getArrRef()),
            builder.getExpression(read.getIndices().front()));
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
        llzk::ensure(writeOp.getIndices().size() == 1,
                     "multidimensional arrays not supported");
        auto arr = writeOp.getArrRef();
        auto index = writeOp.getIndices().front();
        auto value = writeOp.getRvalue();
        if (valueIsSignalRead(arr, tables) || valueIsSignalWrite(arr, tables)) {
          // TODO: Make this behavior configurable
          postcondition.addAntecedent(
              builder.assertEqual(builder.arrayRead(arr, index), value));
          return;
        }
        postcondition.substitute(builder.getConstant(arr),
                                 builder.arrayWrite(arr, index, value));
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
        for (auto invariant : conjecturePredicates(op)) {
          llvm::dbgs() << "; Conjecturing " << invariant.toString() << "\n";
          postcondition.addAntecedent(invariant);
        }
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

cvc5::Term WeakestPreconditionAnalysis::getPostcondition() {

  auto members = structDef.getMemberDefs();
  llzk::ensure(!members.empty(),
               "cannot build postcondition for struct with empty members");

  std::vector<cvc5::Term> memberEquivs;
  for (auto memberDef : members) {
    auto witnessSym = builder.getConstant(memberDef, true);
    auto constraintSym = builder.getConstant(memberDef, false);
    memberEquivs.push_back(builder.assertEqual(witnessSym, constraintSym));
  }

  if (memberEquivs.size() > 1) {
    return mgr.mkTerm(cvc5::Kind::AND, memberEquivs);
  }
  return memberEquivs.front();
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
  os << "(check-sat)\n(get-model)\n";
}

} // namespace lleq
