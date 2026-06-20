/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/StrongestPostcondition.h"
#include "Analysis/ScalarSymbolAnalysis.h"
#include "Verification/VerificationUtils.h"

#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/TypeSwitch.h>

#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/Constrain/IR/Ops.h>
#include <llzk/Dialect/Felt/IR/Ops.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>

#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <unordered_set>
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

cvc5::Term
StrongestPostconditionAnalysis::getCurrentTerm(cvc5::Term term,
                                               const SPState &state) {
  if (auto it = state.bindings.find(term); it != state.bindings.end()) {
    return it->second;
  }
  return term;
}

cvc5::Term
StrongestPostconditionAnalysis::getCurrentValue(mlir::Value value,
                                                const SPState &state) {
  return getCurrentTerm(builder.getConstant(value), state);
}

void StrongestPostconditionAnalysis::bindValue(mlir::Value value, cvc5::Term term,
                                               SPState &state) {
  state.bindings.insert_or_assign(builder.getConstant(value), term);
}

void StrongestPostconditionAnalysis::addConstraint(cvc5::Term term,
                                                   SPState &state) {
  if (state.formula.getKind() == cvc5::Kind::CONSTANT &&
      state.formula.getBooleanValue()) {
    state.formula = term;
    return;
  }
  state.formula = mgr.mkTerm(cvc5::Kind::AND, {state.formula, term});
}

cvc5::Term StrongestPostconditionAnalysis::getExpression(Operation *op,
                                                         const SPState &state) {
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
        llvm::map_to_vector(op->getOperands(), [this, &state](Value value) {
          return getCurrentValue(value, state);
        })};
    return mgr.mkTerm(it->second, {operandTerms.begin(), operandTerms.end()});
  }

  return llvm::TypeSwitch<Operation *, cvc5::Term>(op)
      .Case<MemberReadOp>([this, &state](MemberReadOp read) {
        return getCurrentTerm(
            builder.getConstant(read.getMemberDefOp(tables)->get(),
                                isWitnessOp(read)),
            state);
      })
      .Case<ReadArrayOp>([this, &state](ReadArrayOp read) {
        llzk::ensure(read.getIndices().size() == 1,
                     "multidimensional arrays are not supported");
        return builder.arrayRead(getCurrentValue(read.getArrRef(), state),
                                 getCurrentValue(read.getIndices().front(),
                                                 state));
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
      .Case<array::CreateArrayOp>([this, &state](array::CreateArrayOp createArr) {
        return getCurrentTerm(builder.getConstant(createArr.getResult()), state);
      })
      .Case<llzk::function::CallOp>([this, &state](llzk::function::CallOp call) {
        llzk::ensure(call.calleeIsCompute(),
                     "arbitrary function calls not supported yet");
        auto target = call.getCalleeTarget(tables);
        llzk::ensure(succeeded(target), "failed to resolve callee target");
        auto subcmp = target->get()->getParentOfType<component::StructDefOp>();
        SmallVector<Value> args = llvm::to_vector(call.getArgOperands());
        return builder.initSubcmp(subcmp, args);
      })
      .Default([op](auto) -> cvc5::Term {
        llvm::report_fatal_error("unknown op: " + op->getName().getStringRef());
      });
}

void StrongestPostconditionAnalysis::calculateSP(Operation *op, SPState &state) {
  llvm::TypeSwitch<mlir::Operation *, void>(op)
      .Case<component::CreateStructOp>([](auto) {})
      .Case<MemberWriteOp>([this, &state](MemberWriteOp writeOp) {
        auto member =
            builder.getConstant(writeOp.getMemberDefOp(tables)->get(),
                                isWitnessOp(writeOp));
        auto value = getCurrentValue(writeOp.getVal(), state);
        addConstraint(builder.assertEqual(member, value), state);
        state.bindings.insert_or_assign(member, value);
      })
      .Case<WriteArrayOp>([this, &state](WriteArrayOp writeOp) {
        llzk::ensure(writeOp.getIndices().size() == 1,
                     "multidimensional arrays not supported");
        auto arr = writeOp.getArrRef();
        auto index = getCurrentValue(writeOp.getIndices().front(), state);
        auto value = getCurrentValue(writeOp.getRvalue(), state);
        auto currentArr = getCurrentValue(arr, state);
        if (valueIsSignalRead(arr, tables) || valueIsSignalWrite(arr, tables)) {
          addConstraint(
              builder.assertEqual(builder.arrayRead(currentArr, index), value),
              state);
          return;
        }
        state.bindings.insert_or_assign(builder.getConstant(arr),
                                        builder.arrayWrite(currentArr, index,
                                                           value));
      })
      .Case<constrain::EmitEqualityOp>([this, &state](EmitEqualityOp eqOp) {
        addConstraint(
            builder.assertEqual(getCurrentValue(eqOp.getLhs(), state),
                                getCurrentValue(eqOp.getRhs(), state)),
            state);
      })
      .Case<scf::IfOp>([this, &state](scf::IfOp ifOp) { calculateSP(ifOp, state); })
      .Case<llzk::function::CallOp>([this, &state](llzk::function::CallOp call) {
        if (call.calleeIsConstrain()) {
          auto target = call.getCalleeTarget(tables);
          llzk::ensure(succeeded(target), "failed to resolve callee target");
          auto subcmpVal = getCurrentValue(call.getArgOperands().front(), state);
          auto subcmp = target->get()->getParentOfType<component::StructDefOp>();
          SmallVector<Value> args =
              llvm::to_vector(call.getArgOperands().drop_front());
          addConstraint(
              builder.assertEqual(subcmpVal, builder.initSubcmp(subcmp, args)),
              state);
          return;
        }

        auto expression = getExpression(call, state);
        bindValue(call->getResult(0), expression, state);
      })
      .Default([this, &state](auto op) {
        if (op->getNumResults() == 0) {
          return;
        }
        auto expression = getExpression(op, state);
        bindValue(op->getResult(0), expression, state);
      });
}

void StrongestPostconditionAnalysis::calculateSP(Block *block, SPState &state) {
  for (auto &op : *block) {
    if (&op == block->getTerminator()) {
      continue;
    }
    calculateSP(&op, state);
  }
}

void StrongestPostconditionAnalysis::calculateSP(mlir::scf::IfOp ifOp,
                                                 SPState &state) {
  llzk::ensure(ifOp.getElseRegion().hasOneBlock(),
               "expected scf.if to have an else region");
  auto condition = getCurrentValue(ifOp.getCondition(), state);
  auto notCondition = mgr.mkTerm(cvc5::Kind::NOT, {condition});

  SPState thenState{state};
  SPState elseState{state};
  calculateSP(&ifOp.getThenRegion().front(), thenState);
  calculateSP(&ifOp.getElseRegion().front(), elseState);

  auto thenFormula =
      mgr.mkTerm(cvc5::Kind::AND, {condition, thenState.formula});
  auto elseFormula =
      mgr.mkTerm(cvc5::Kind::AND, {notCondition, elseState.formula});
  state.formula = mgr.mkTerm(cvc5::Kind::OR, {thenFormula, elseFormula});

  std::unordered_set<cvc5::Term, std::hash<cvc5::Term>> keys;
  for (const auto &[term, _] : state.bindings) {
    keys.insert(term);
  }
  for (const auto &[term, _] : thenState.bindings) {
    keys.insert(term);
  }
  for (const auto &[term, _] : elseState.bindings) {
    keys.insert(term);
  }

  for (auto key : keys) {
    auto thenIt = thenState.bindings.find(key);
    auto elseIt = elseState.bindings.find(key);
    auto thenValue = thenIt == thenState.bindings.end() ? key : thenIt->second;
    auto elseValue = elseIt == elseState.bindings.end() ? key : elseIt->second;
    if (thenValue == elseValue) {
      if (thenValue != key) {
        state.bindings.insert_or_assign(key, thenValue);
      } else {
        state.bindings.erase(key);
      }
      continue;
    }
    auto merged = mgr.mkTerm(cvc5::Kind::ITE, {condition, thenValue, elseValue});
    state.bindings.insert_or_assign(key, merged);
  }
}

void StrongestPostconditionAnalysis::populateVerificationConditions() {
  llzk::ensure(succeeded(ensureProductFunc(
                   structDef->getParentOfType<ModuleOp>(), structDef)),
               "failed to align product func");
  SPState state{mgr.mkBoolean(true), {}};
  calculateSP(&structDef.getProductFuncOp().getFunctionBody().front(), state);

  verificationConditions = state.formula;
  extraDecls = builder.getExtraDecls(verificationConditions);
  declBounds = builder.getDeclBounds(extraDecls, field.prime());
}

cvc5::Term StrongestPostconditionAnalysis::generateVerificationConditions() {
  llzk::ensure(succeeded(ensureProductFunc(
                   structDef->getParentOfType<ModuleOp>(), structDef)),
               "failed to align product func");
  SPState state{mgr.mkBoolean(true), {}};
  calculateSP(&structDef.getProductFuncOp().getFunctionBody().front(), state);

  return state.formula;
}

void StrongestPostconditionAnalysis::emit(llvm::raw_ostream &os) {
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

  os << "; Strongest postcondition\n";
  os << "(assert " << verificationConditions.toString() << ")\n";
  os << "(check-sat)\n(get-model)\n";
}

} // namespace lleq
