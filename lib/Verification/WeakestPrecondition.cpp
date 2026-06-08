/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/WeakestPrecondition.h"
#include "Analysis/ScalarSymbolAnalysis.h"
#include "Verification/TermUtils.h"
#include "Verification/VerificationUtils.h"

#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/TypeSwitch.h>

#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/Constrain/IR/Ops.h>
#include <llzk/Dialect/Felt/IR/Ops.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Util/DynamicAPIntHelper.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>

#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
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

// TODO: I *think* its enough to implement subcmp calls to @compute/@constrain
// in here since writing to the subcmp member should handle the assertion, and:
// (1) If the top struct was aligned mechanically the SSA values being written
// to _w and _c should be distinct, and
// (2) Otherwise if they aren't distinct, it should still be correct to assert
// these two (init-...) invocations equal?
cvc5::Term WeakestPreconditionAnalysis::getExpression(Operation *op) {
  static llvm::DenseMap<StringRef, cvc5::Kind> opToTermKind = {
      {"felt.add", cvc5::Kind::ADD},
      {"felt.sub", cvc5::Kind::SUB},
      {"felt.mul", cvc5::Kind::MULT},
      {"felt.smod", cvc5::Kind::INTS_MODULUS},
      {"felt.sintdiv", cvc5::Kind::INTS_DIVISION}};

  if (auto it = opToTermKind.find(op->getName().getStringRef());
      it != opToTermKind.end()) {
    SmallVector<cvc5::Term> operandTerms{
        llvm::map_to_vector(op->getOperands(), [this](Value value) {
          return builder.getConstant(value);
        })};
    return mgr.mkTerm(it->second, {operandTerms.begin(), operandTerms.end()});
  }

  return llvm::TypeSwitch<Operation *, cvc5::Term>(op)
      .Case<MemberReadOp>([this](MemberReadOp read) {
        return builder.getConstant(read.getMemberDefOp(tables)->get(),
                                   isWitnessOp(read));
      })
      .Case<ReadArrayOp>([this](ReadArrayOp read) {
        llzk::ensure(read.getIndices().size() == 1,
                     "multidimensional arrays are not supported");
        return builder.arrayRead(read.getArrRef(), read.getIndices().front());
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
      .Default([op](auto) -> cvc5::Term {
        llvm::report_fatal_error("unknown op: " + op->getName().getStringRef());
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
    return memberDef->get().getSignal();
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
      return memberDef->get().getSignal();
    }
  }
  return false;
}

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
          postcondition.addAntecedent(
              builder.assertEqual(builder.arrayRead(arr, index), value));
          return;
        }
        postcondition.substitute(builder.getConstant(writeOp.getArrRef()),
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
          postcondition.substitute(builder.getConstant(call->getResult(0)),
                                   expression);
        }
      })
      .Default([this, &postcondition](auto op) {
        auto expression = getExpression(op);
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

std::pair<cvc5::Term, TermBuilder::TermSet>
WeakestPreconditionAnalysis::generateVerificationConditions() {
  llzk::ensure(succeeded(ensureProductFunc(
                   structDef->getParentOfType<ModuleOp>(), structDef)),
               "failed to align product func");
  auto postcondition = ConjunctionTerm::of(getPostcondition());
  calculateWP(&structDef.getProductFuncOp().getFunctionBody().front(),
              postcondition);

  auto term = postcondition.buildTerm(mgr);
  auto extraDecls = builder.getExtraDecls(term);
  return {term, extraDecls};
}

} // namespace lleq
