/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/WeakestPrecondition.h"
#include "Analysis/ScalarSymbolAnalysis.h"
#include "Verification/TermUtils.h"
#include "Verification/VerificationUtils.h"

#include <complex>
#include <fstream>
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
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  auto tempStdin = std::tmpnam(nullptr);
  auto tempStdout = std::tmpnam(nullptr);
#pragma clang diagnostic pop

  std::ofstream os{tempStdin};
  os << query.data();
  os.close();

  auto solverPath = resolveZ3Path();
  SmallVector<StringRef> args{"z3", "-in", "-smt2"};

  std::string error;
  auto code = llvm::sys::ExecuteAndWait(
      solverPath, args,
      /*Env=*/std::nullopt,
      /*Redirects=*/
      {std::string{tempStdin}, std::string{tempStdout}, ""}, 0, 0, &error);
  if (code) {
    llvm::report_fatal_error(error.c_str());
  }

  std::ifstream is{tempStdout};
  std::string result;
  is >> result;
  return result == "unsat";
}

Block *nestedLoopBody(scf::ForOp loop, SmallVector<LoopCounterInfo> &loopInfo,
                      TermBuilder &builder) {
  loopInfo.push_back(
      LoopCounterInfo{builder.getExpression(loop.getInductionVar()),
                      Range{builder.getExpression(loop.getLowerBound()),
                            builder.getExpression(loop.getUpperBound()),
                            builder.getExpression(loop.getStep())}});
  if (auto first = dyn_cast<scf::ForOp>(loop.getBody()->front())) {
    return nestedLoopBody(first, loopInfo, builder);
  }
  return loop.getBody();
}

struct FailingCore {
  SmallVector<cvc5::Term> terms;
  SmallVector<Annotation> annotations;
};

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

  llzk::ensure(loopInfo.size() == 1, "nested loop support in-progress");

  auto [i, range] = loopInfo.front();
  auto [lb, ub, step] = range;

  SmallVector<cvc5::Term> predicates;
  SmallVector<int64_t> arraySizes;
  for (auto [write, index] : witnessWrites) {
    auto memberDef = write.getMemberDefOp(tables)->get();
    auto arr_w_i = builder.arrayRead(builder.getConstant(memberDef, true), i);
    auto arr_c_i = builder.arrayRead(builder.getConstant(memberDef, false), i);
    predicates.push_back(builder.assertEqual(arr_w_i, arr_c_i));
    // We've already asserted that the array is not multidimensional
    arraySizes.push_back(
        dyn_cast<array::ArrayType>(memberDef.getType()).getShape().front());
  }

  // Turn `predicate(var)` into `forall x in range(lb, ub, step), predicate(x)`
  auto quantifyPredicate = [this](cvc5::Term predicate, cvc5::Term var,
                                  cvc5::Term lb, cvc5::Term ub,
                                  cvc5::Term step) {
    auto x = mgr.mkVar(mgr.getIntegerSort(), "x");
    auto boundPos = mgr.mkTerm(cvc5::Kind::LEQ, {lb, x});
    auto boundInv = mgr.mkTerm(cvc5::Kind::LT, {x, ub});

    auto boundStep =
        mgr.mkTerm(cvc5::Kind::EQUAL,
                   {mgr.mkTerm(cvc5::Kind::INTS_MODULUS,
                               {mgr.mkTerm(cvc5::Kind::SUB, {x, lb}), step}),
                    builder.getInteger(0)});
    auto bound = mgr.mkTerm(cvc5::Kind::AND, {boundPos, boundInv, boundStep});

    return mgr.mkTerm(cvc5::Kind::FORALL,
                      {mgr.mkTerm(cvc5::Kind::VARIABLE_LIST, {x}),
                       mgr.mkTerm(cvc5::Kind::IMPLIES,
                                  {bound, predicate.substitute(var, x)})});
  };

  // Filter out which predicates are inductive
  SmallVector<cvc5::Term> inductivePredicates;
  // Conjuncts necessary to strengthen the inductive invariant
  // SmallVector<cvc5::Term> strengthenings;
  // forall x in range(lb, i, step), forall sig, sig_c[x] == sig_w[x]
  cvc5::Term bodyPrecondition =
      quantifyPredicate(conjunctAll(predicates, mgr), i, lb, i, step);
  cvc5::Term nextI = mgr.mkTerm(cvc5::Kind::ADD, {i, step});

  SmallVector<cvc5::Term> sliceAssertions;
  for (auto [size, predicate] : llvm::zip(arraySizes, predicates)) {
    // We can strengthen the invariant to say the array is equal outside the
    // slice visited by the for loop as well (note: this isn't quite right if,
    // e.g., the loop isn't a basic "step 1, write to arr[i]", but its pretty
    // hard to do much better in general)
    sliceAssertions.push_back(
        quantifyPredicate(predicate, i, builder.getInteger(0), lb, step));
    sliceAssertions.push_back(
        quantifyPredicate(predicate, i, ub, builder.getInteger(size), step));
  }

  auto strengthened = conjunctAll(sliceAssertions, mgr);
  auto strengthenedPrecondition =
      conjunctAll({bodyPrecondition, strengthened}, mgr);

  for (auto [size, predicate] : llvm::zip(arraySizes, predicates)) {
    auto postcondition =
        ConjunctionTerm::of(quantifyPredicate(predicate, i, lb, nextI, step));

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
    llvm::dbgs() << query << "\n";
    if (checkUnsatWithZ3(query)) {
      inductivePredicates.push_back(predicate);
      // TODO: Also add annotations for the strengthenings
    }
  }

  // Check that the resulting predicate entails the postcondition
  auto inductiveInvariant = conjunctAll(inductivePredicates, mgr);
  inductiveInvariant = quantifyPredicate(inductiveInvariant, i, lb, ub, step);

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
      if (!ann.isArray) {
        sliceAssertions.push_back(elem);
      } else {
        return failure();
      }
    }
    return conjunctAll(sliceAssertions, mgr);
  }

  // TODO: add strengthenings
}

TermBuilder::TermSet
WeakestPreconditionAnalysis::conjecturePredicates(mlir::scf::ForOp loop) {
  SmallVector<LoopCounterInfo> loopInfo;
  auto *body = nestedLoopBody(loop, loopInfo, builder);

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

  llzk::ensure(loopInfo.size() == 1, "nested loop support in-progress");

  std::vector<cvc5::Term> predicates;
  for (auto [write, index] : witnessWrites) {
    auto arr_w_i = builder.arrayRead(
        builder.getConstant(write.getMemberDefOp(tables)->get(), true),
        builder.getExpression(index));
    auto arr_c_i = builder.arrayRead(
        builder.getConstant(write.getMemberDefOp(tables)->get(), false),
        builder.getExpression(index));
    predicates.push_back(builder.assertEqual(arr_w_i, arr_c_i));
  }

  TermBuilder::TermSet invariants;
  auto [i, range] = loopInfo.front();
  auto [lb, ub, step] = range;

  // all signals are equal at i
  auto equalAtI = predicates.size() == 1
                      ? predicates.front()
                      : mgr.mkTerm(cvc5::Kind::AND, predicates);

  // Turn `P(i)` into `forall lb <= x < ub and step | (x - lb), P(x)`
  auto quantifyPredicate = [this](cvc5::Term predicate, cvc5::Term var,
                                  cvc5::Term ub, cvc5::Term lb,
                                  cvc5::Term step) {
    auto x = mgr.mkVar(mgr.getIntegerSort(), "x");
    auto boundPos = mgr.mkTerm(cvc5::Kind::LEQ, {lb, x});
    auto boundInv = mgr.mkTerm(cvc5::Kind::LT, {x, ub});

    auto boundStep =
        mgr.mkTerm(cvc5::Kind::EQUAL,
                   {mgr.mkTerm(cvc5::Kind::INTS_MODULUS,
                               {mgr.mkTerm(cvc5::Kind::SUB, {x, lb}), step}),
                    builder.getInteger(0)});
    auto bound = mgr.mkTerm(cvc5::Kind::AND, {boundPos, boundInv, boundStep});

    return mgr.mkTerm(cvc5::Kind::FORALL,
                      {mgr.mkTerm(cvc5::Kind::VARIABLE_LIST, {x}),
                       mgr.mkTerm(cvc5::Kind::IMPLIES,
                                  {bound, predicate.substitute(var, x)})});
  };

  // Turn "all signals equal at i" to "all signals equal at all x < i"
  auto allEqualUpToCurrent = quantifyPredicate(equalAtI, i, i, lb, step);

  // Now, filter out the ones that aren't loop invariants
  for (auto invariant : predicates) {
    // P(i) is an invariant if:
    //  {C /\ forall x < i, P(x)} B {forall x < i + step, P(x)}
    // holds
    auto invariantHeld =
        mgr.mkTerm(cvc5::Kind::AND,
                   {mgr.mkTerm(cvc5::Kind::LT, {i, ub}), allEqualUpToCurrent});

    // We need to say something about intermediate signals that show up in the
    // loop body but won't be proven equivalent by the invariant (either because
    // they're scalar or because they aren't visited by the loop). For now, just
    // do something stupid and assert equality for all array elements that come
    // before the loop.
    auto extraAssertions =
        quantifyPredicate(equalAtI, i, lb, builder.getInteger(0), step);

    // Calculate wp(B, I)
    auto precondition = ConjunctionTerm::of(quantifyPredicate(
        invariant, i, mgr.mkTerm(cvc5::Kind::ADD, {i, step}), lb, step));
    precondition.addAntecedent(extraAssertions);
    calculateWP(body, precondition);

    // Check C /\ AllI => wp(B, I)
    auto isInvariant = mgr.mkTerm(cvc5::Kind::IMPLIES,
                                  {invariantHeld, precondition.buildTerm(mgr)});

    auto query = buildSMTQuery(isInvariant, builder, field);
    llvm::dbgs() << invariant.toString() << "\n";
    llvm::dbgs() << query << "\n";
    if (checkUnsatWithZ3(query)) {
      // Assume the invariant holds after exiting the loop
      invariants.insert(quantifyPredicate(invariant, i, ub, lb, step));
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
      llzk::ensure(arrType.getShape().size() == 1,
                   "multidimensional arrays not yet supported");

      auto size = arrType.getShape().front();
      annotations.push_back(
          Annotation{true,
                     {{builder.getInteger(0), builder.getInteger(size),
                       builder.getInteger(1)}}});
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
