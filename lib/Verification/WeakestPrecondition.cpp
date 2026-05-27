/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/WeakestPrecondition.h"
#include "Analysis/ScalarSymbolAnalysis.h"
#include "Verification/VerificationUtils.h"

#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/TypeSwitch.h>

#include <llvm/Support/ErrorHandling.h>
#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/Constrain/IR/Ops.h>
#include <llzk/Dialect/Felt/IR/Ops.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
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

cvc5::Term WeakestPreconditionAnalysis::getExpression(Operation *op) {
  static llvm::StringMap<cvc5::Kind> opToTermKind = {
      {"felt.add", cvc5::Kind::ADD},
      {"felt.sub", cvc5::Kind::SUB},
      {"felt.mul", cvc5::Kind::MULT}};

  if (auto it = opToTermKind.find(op->getName().getStringRef());
      it != opToTermKind.end()) {
    SmallVector<cvc5::Term> operandTerms{llvm::map_to_vector(
        op->getOperands(), [this](Value value) { return getConstant(value); })};
    return mgr.mkTerm(it->second, {operandTerms.begin(), operandTerms.end()});
  }

  return llvm::TypeSwitch<Operation *, cvc5::Term>(op)
      .Case<MemberReadOp>([this](MemberReadOp read) {
        return getConstant(read.getMemberName(), isWitnessOp(read));
      })
      .Case<ReadArrayOp>([this](ReadArrayOp read) {
        llzk::ensure(read.getIndices().size() == 1,
                     "multidimensional arrays are not supported");
        return mgr.mkTerm(cvc5::Kind::SELECT,
                          {getConstant(read.getArrRef()),
                           getConstant(read.getIndices().front())});
      })
      .Case<FeltConstantOp>([this](FeltConstantOp constOp) {
        SmallString<64> str;
        constOp.getValue().getValue().toStringUnsigned(str);
        return mgr.mkInteger(std::string{str});
      })
      .Default([op](auto) -> cvc5::Term {
        llvm::report_fatal_error("unknown op: " + op->getName().getStringRef());
      });
}

cvc5::Term WeakestPreconditionAnalysis::getConstant(mlir::Value value) {
  static DenseMap<Value, cvc5::Term> constants;
  if (auto it = constants.find(value); it != constants.end()) {
    return it->second;
  }

  std::optional<std::string> name;
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    name.emplace("arg" + std::to_string(blockArg.getArgNumber()));
  }

  auto newConst = mgr.mkConst(mgr.getIntegerSort(), name);
  constants.insert({value, newConst});
  return newConst;
}

cvc5::Term WeakestPreconditionAnalysis::getConstant(mlir::StringRef memberName,
                                                    bool isWitness) {
  static llvm::StringMap<cvc5::Term> witnessMembers;
  static llvm::StringMap<cvc5::Term> constrainMembers;
  auto &memberMap = isWitness ? witnessMembers : constrainMembers;

  if (auto it = memberMap.find(memberName); it != memberMap.end()) {
    return it->second;
  }
  auto newTerm = mgr.mkConst(mgr.getIntegerSort(),
                             (memberName + (isWitness ? "_w" : "_c")).str());
  memberMap.insert({memberName, newTerm});
  return newTerm;
}

cvc5::Term WeakestPreconditionAnalysis::calculateWP(Operation *op,
                                                    cvc5::Term postcondition) {
  auto precondition =
      llvm::TypeSwitch<mlir::Operation *, cvc5::Term>(op)
          .Case<component::CreateStructOp>(
              [&postcondition](auto) { return postcondition; })
          .Case<MemberWriteOp>([this, &postcondition](MemberWriteOp writeOp) {
            auto assertion = mgr.mkTerm(cvc5::Kind::EQUAL,
                                        {getConstant(writeOp.getMemberName(),
                                                     /*isWitness=*/true),
                                         getConstant(writeOp.getVal())});
            return mgr.mkTerm(cvc5::Kind::IMPLIES, {assertion, postcondition});
          })
          .Case<WriteArrayOp>([this, &postcondition](WriteArrayOp writeOp) {
            llzk::ensure(writeOp.getIndices().size() == 1,
                         "multidimensional arrays not supported");
            auto writeExpr = mgr.mkTerm(
                cvc5::Kind::STORE, {getConstant(writeOp.getArrRef()),
                                    getConstant(writeOp.getIndices().front()),
                                    getConstant(writeOp.getRvalue())});
            return postcondition.substitute(getConstant(writeOp.getArrRef()),
                                            writeExpr);
          })
          .Case<constrain::EmitEqualityOp>([this, &postcondition](
                                               EmitEqualityOp eqOp) {
            auto assertion =
                mgr.mkTerm(cvc5::Kind::EQUAL, {getConstant(eqOp.getLhs()),
                                               getConstant(eqOp.getRhs())});
            return mgr.mkTerm(cvc5::Kind::IMPLIES, {assertion, postcondition});
          })
          .Default([this, &postcondition](auto op) {
            auto expression = getExpression(op);
            return postcondition.substitute(getConstant(op->getResult(0)),
                                            expression);
          });
  return precondition;
}
cvc5::Term WeakestPreconditionAnalysis::calculateWP(Block *block,
                                                    cvc5::Term postcondition) {
  for (auto &op : llvm::iterator_range(block->rbegin(), block->rend())) {
    if (&op == block->getTerminator()) {
      continue;
    }
    postcondition = calculateWP(&op, postcondition);
  }
  return postcondition;
}

cvc5::Term getPostcondition(component::StructDefOp structDef,
                            cvc5::TermManager &mgr) {

  auto members = structDef.getMemberDefs();
  std::vector<cvc5::Term> memberEquivs;

  for (auto memberDef : members) {
    auto memberName = memberDef.getSymName();
    auto witnessSym =
        mgr.mkConst(mgr.getIntegerSort(), (memberName + "_w").str());
    auto constraintSym =
        mgr.mkConst(mgr.getIntegerSort(), (memberName + "_c").str());
    memberEquivs.push_back(
        mgr.mkTerm(cvc5::Kind::EQUAL, {witnessSym, constraintSym}));
  }

  return mgr.mkTerm(cvc5::Kind::AND, memberEquivs);
}

cvc5::Term WeakestPreconditionAnalysis::generateVerificationConditions() {
  llzk::ensure(succeeded(ensureProductFunc(
                   structDef->getParentOfType<ModuleOp>(), structDef)),
               "failed to align product func");
  return calculateWP(&structDef.getProductFuncOp().getFunctionBody().front(),
                     getPostcondition(structDef, mgr));
}

} // namespace lleq
