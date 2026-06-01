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
        return builder.getConstant(read.getMemberName(), isWitnessOp(read));
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
      .Default([op](auto) -> cvc5::Term {
        llvm::report_fatal_error("unknown op: " + op->getName().getStringRef());
      });
}

void WeakestPreconditionAnalysis::calculateWP(Operation *op,
                                              ImplicationTerm &postcondition) {
  llvm::TypeSwitch<mlir::Operation *, void>(op)
      .Case<component::CreateStructOp>(
          [&postcondition](auto) { return postcondition; })
      .Case<MemberWriteOp>([this, &postcondition](MemberWriteOp writeOp) {
        postcondition.addAntecedent(builder.assertEqual(
            builder.getConstant(writeOp.getMemberName(), /*isWitness=*/true),
            writeOp.getVal()));
      })
      .Case<WriteArrayOp>([this, &postcondition](WriteArrayOp writeOp) {
        llzk::ensure(writeOp.getIndices().size() == 1,
                     "multidimensional arrays not supported");
        postcondition.substitute(
            builder.getConstant(writeOp.getArrRef()),
            builder.arrayWrite(writeOp.getArrRef(),
                               writeOp.getIndices().front(),
                               writeOp.getRvalue()));
      })
      .Case<constrain::EmitEqualityOp>(
          [this, &postcondition](EmitEqualityOp eqOp) {
            postcondition.addAntecedent(
                builder.assertEqual(eqOp.getLhs(), eqOp.getRhs()));
          })
      .Case<scf::IfOp>([this, &postcondition](scf::IfOp op) {
        calculateWP(op, postcondition);
      })
      .Default([this, &postcondition](auto op) {
        auto expression = getExpression(op);
        postcondition.substitute(builder.getConstant(op->getResult(0)),
                                 expression);
      });
}

void WeakestPreconditionAnalysis::calculateWP(Block *block,
                                              ImplicationTerm &postcondition) {
  // TODO: also return yielded values
  for (auto &op : llvm::iterator_range(block->rbegin(), block->rend())) {
    if (&op == block->getTerminator()) {
      continue;
    }
    calculateWP(&op, postcondition);
  }
}

void WeakestPreconditionAnalysis::calculateWP(mlir::scf::IfOp ifOp,
                                              ImplicationTerm &postcondition) {}

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
  auto postcondition = ImplicationTerm::of(getPostcondition(structDef, mgr));
  calculateWP(&structDef.getProductFuncOp().getFunctionBody().front(),
              postcondition);
  return postcondition.buildTerm(mgr);
}

} // namespace lleq
