/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/SMTLIBEquivalenceEmitter.h"

#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Analysis/LightweightSignalEquivalenceAnalysis.h>
#include <llzk/Transforms/LLZKComputeConstrainToProductPass.h>
#include <llzk/Util/SymbolHelper.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LLVM.h>
#include <string>
#include <utility>

#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/SMT/IR/SMTOps.h>
#include <llzk/Dialect/Struct/IR/Ops.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace mlir;
using namespace llzk;

namespace llzk::smt {
std::unique_ptr<mlir::Pass> createSMTLoweringPass();
std::unique_ptr<mlir::Pass> createSMTCFLoweringPass();
} // namespace llzk::smt

namespace {

std::string sanitizeSymbol(llvm::StringRef name) {
  std::string out;
  out.reserve(name.size());
  for (char ch : name) {
    if (llvm::isAlnum(ch) || ch == '_' || ch == '.' || ch == '$') {
      out.push_back(ch);
    } else {
      out.push_back('_');
    }
  }
  if (out.empty()) {
    out = "tmp";
  }
  return out;
}

[[noreturn]] void crash() { llvm::report_fatal_error("SMT emission failed"); }

} // namespace
class SMTLIBFunctionEmitter {
public:
  explicit SMTLIBFunctionEmitter(raw_ostream &os) : os(os) {}

  LogicalResult emit(func::FuncOp func) {
    // Can probably do better here
    os << "(set-logic ALL)\n";
    for (auto [index, arg] : llvm::enumerate(func.getArguments())) {
      std::string name = "arg" + std::to_string(index);
      values[arg] = name;
      os << "(declare-const " << name << " " << sortForType(arg.getType())
         << ")\n";
    }

    if (failed(processBlock(func.getBody().front()))) {
      return failure();
    }

    return success();
  }

private:
  LogicalResult processBlock(Block &block) {
    for (Operation &op : block.without_terminator()) {

      LogicalResult result =
          TypeSwitch<Operation *, LogicalResult>(&op)
              .Case<scf::IfOp>([](scf::IfOp ifOp) {
                ifOp.emitError() << "control flow not supported in SMT emitter";
                return failure();
              })
              .Case<smt::DeclareFunOp>([this](auto declFunOp) {
                return processDeclareFun(declFunOp);
              })
              .Case<smt::AssertOp>(
                  [this](auto assertOp) { return processAssert(assertOp); })
              .Case<UnrealizedConversionCastOp>(
                  [this](auto castOp) { return processUnrealizedCast(castOp); })
              .Case<arith::ConstantOp>([this](arith::ConstantOp constOp) {
                values[constOp.getResult()] = printArithConstant(constOp);
                return success();
              })
              .Default([this](auto *op) {
                if (op->getNumResults() == 1) {
                  FailureOr<std::string> expr = buildExpression(op);
                  if (failed(expr)) {
                    return failure();
                  }
                  values[op->getResult(0)] = std::move(*expr);
                  return success();
                }
                op->emitError()
                    << "unsupported operation while emitting SMTLIB "
                       "equivalence query";
                return failure();
              });

      if (failed(result)) {
        return failure();
      }
    }

    return success();
  }

  LogicalResult processDeclareFun(smt::DeclareFunOp &op) {
    auto namePrefix = op.getNamePrefix();
    std::string symbol = namePrefix ? sanitizeSymbol(*namePrefix)
                                    : ("tmp" + std::to_string(nextTempId++));

    values[op.getResult()] = symbol;

    os << "(declare-fun " << symbol << " () " << sortForType(op.getType())
       << ")\n";
    return success();
  }

  LogicalResult processAssert(smt::AssertOp &op) {
    FailureOr<std::string> input = lookup(op.getInput());
    if (failed(input)) {
      return op.emitError() << "missing SMT expression for assertion input";
    }

    os << "(assert " << *input << ")\n";
    return success();
  }

  LogicalResult processUnrealizedCast(UnrealizedConversionCastOp op) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
      return op.emitError()
             << "only one-to-one unrealized casts are supported in SMTLIB "
                "emission";
    }

    FailureOr<std::string> input = lookup(op.getOperand(0));
    if (failed(input)) {
      return op.emitError() << "missing expression for conversion input";
    }
    values[op.getResult(0)] = *input;
    return success();
  }

  FailureOr<std::string> buildExpression(Operation *op) {

    static DenseMap<StringRef, StringRef> opToSmtOp = {
        {"smt.int.neg", "-"},      {"smt.not", "not"},
        {"smt.int.add", "+"},      {"smt.int.mul", "*"},
        {"smt.int.sub", "-"},      {"smt.int.mod", "mod"},
        {"smt.eq", "="},           {"smt.int.and", "and"},
        {"smt.int.or", "or"},      {"smt.int.xor", "xor"},
        {"smt.int.implies", "=>"}, {"smt.int.not", "not"},
        {"smt.ite", "ite"},
    };

    return TypeSwitch<Operation *, FailureOr<std::string>>(op)
        .Case<smt::IntConstantOp>([](smt::IntConstantOp constOp) {
          SmallString<32> str;
          constOp.getValue().toStringSigned(str);
          return str.str().str();
        })
        .Case<smt::BoolConstantOp>([](smt::BoolConstantOp constOp) {
          return success(constOp.getValue() ? "true" : "false");
        })
        .Case<smt::IntCmpOp>([this](smt::IntCmpOp cmpOp) {
          std::string smtOp;
          switch (cmpOp.getPred()) {
          case smt::IntPredicate::lt:
            smtOp = "<";
            break;
          case smt::IntPredicate::le:
            smtOp = "<=";
            break;
          case smt::IntPredicate::gt:
            smtOp = ">";
            break;
          case smt::IntPredicate::ge:
            smtOp = ">=";
            break;
          }
          return buildSExpr(smtOp, cmpOp.getOperands());
        })
        .Default([this, op](auto) {
          if (auto it = opToSmtOp.find(op->getName().getStringRef());
              it != opToSmtOp.end()) {
            return buildSExpr(it->second, op->getOperands());
          }
          op->emitError() << "unsupported expression op";
          crash();
        });

    return failure();
  }

  FailureOr<std::string> buildSExpr(llvm::StringRef opName, ValueRange inputs) {
    std::string expr = "(" + opName.str();
    for (Value input : inputs) {
      FailureOr<std::string> value = lookup(input);
      if (failed(value)) {
        return failure();
      }
      expr += " " + *value;
    }
    expr += ")";
    return expr;
  }

  FailureOr<std::string> lookup(Value value) { return lookup(value, values); }

  FailureOr<std::string>
  lookup(Value value, const llvm::DenseMap<Value, std::string> &state) {
    auto it = state.find(value);
    if (it == state.end()) {
      return failure();
    }
    return it->second;
  }

  std::string printArithConstant(arith::ConstantOp constOp) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      if (intAttr.getType().isInteger(1)) {
        return intAttr.getInt() == 0 ? "false" : "true";
      }
      SmallString<32> constant;
      intAttr.getAPSInt().toStringSigned(constant);
      return constant.str().str();
    }
    constOp->emitOpError("unsupported arith.constant");
    crash();
  }

  std::string sortForType(Type type) {
    return TypeSwitch<Type, std::string>(type)
        .Case<smt::IntType>([](auto) { return "Int"; })
        .Case<smt::BoolType>([](auto) { return "Bool"; })
        .Default([](Type type) {
          if (type.isInteger(1)) {
            return "Bool";
          }
          llvm::report_fatal_error("unsupported SMT sort");
        });
  }

  llvm::raw_ostream &os;
  llvm::DenseMap<Value, std::string> values;

  unsigned nextTempId = 0;
};

LogicalResult ensureProductFunc(ModuleOp module,
                                component::StructDefOp structDef) {
  if (structDef.getProductFuncOp()) {
    return success();
  }

  auto computeFunc = structDef.getComputeFuncOp();
  auto constrainFunc = structDef.getConstrainFuncOp();
  if (!computeFunc || !constrainFunc) {
    return structDef.emitError()
           << "expected the selected struct to define either @product or both "
              "@compute and @constrain";
  }

  SymbolTableCollection tables;
  LightweightSignalEquivalenceAnalysis equivalence(module);
  ProductAligner aligner(tables, equivalence);
  auto productFunc = aligner.alignFuncs(structDef, computeFunc, constrainFunc);
  if (!productFunc) {
    return structDef.emitError()
           << "failed to align @compute/@constrain into @product";
  }

  return aligner.alignCalls(productFunc);
}

FailureOr<func::FuncOp> lowerToSMT(component::StructDefOp structDef,
                                   llvm::StringRef fieldName) {
  auto *ctx = structDef.getContext();
  auto module = structDef->getParentOfType<ModuleOp>();
  if (!module) {
    return failure();
  }

  auto structRef = structDef.getFullyQualifiedName();

  auto cloned = cast<ModuleOp>(module->clone());
  auto clonedStruct = cloned.lookupSymbol<component::StructDefOp>(structRef);
  if (!clonedStruct) {
    cloned.emitError() << "selected struct disappeared while cloning module";
    return failure();
  }

  if (failed(ensureProductFunc(cloned, clonedStruct))) {
    return failure();
  }

  auto smtPass = llzk::smt::createSMTLoweringPass();
  std::string options = ("field=" + fieldName).str();
  if (failed(smtPass->initializeOptions(options, [&](const llvm::Twine &err) {
        return cloned.emitError() << err;
      }))) {
    return failure();
  }

  PassManager nonCFPM(ctx, PassManager::getAnyOpAnchorName(),
                      PassManager::Nesting::Implicit);
  nonCFPM.enableVerifier(false);
  nonCFPM.addPass(std::move(smtPass));
  nonCFPM.addPass(createCanonicalizerPass());
  nonCFPM.addPass(createCSEPass());

  if (failed(nonCFPM.run(cloned))) {
    return failure();
  }

  SmallVector<scf::IfOp> ifOps;
  cloned.walk([&](scf::IfOp ifOp) { ifOps.push_back(ifOp); });

  auto cloneBranch = [&](OpBuilder &builder, Block &block,
                         IRMapping &mapping) -> FailureOr<Value> {
    for (Operation &op : block.without_terminator()) {
      builder.clone(op, mapping);
    }

    auto yieldOp = dyn_cast<scf::YieldOp>(block.getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
      return failure();
    }

    return mapping.lookupOrDefault(yieldOp.getOperand(0));
  };

  for (scf::IfOp ifOp : llvm::reverse(ifOps)) {
    if (ifOp.getNumResults() != 1 || ifOp.getElseRegion().empty()) {
      return ifOp.emitError()
             << "unsupported scf.if shape while preparing SMT emission";
    }

    OpBuilder builder(ifOp);
    IRMapping thenMapping;
    IRMapping elseMapping;

    FailureOr<Value> thenValue =
        cloneBranch(builder, ifOp.getThenRegion().front(), thenMapping);
    FailureOr<Value> elseValue =
        cloneBranch(builder, ifOp.getElseRegion().front(), elseMapping);
    if (failed(thenValue) || failed(elseValue)) {
      return ifOp.emitError()
             << "expected each scf.if branch to yield exactly one value";
    }

    Value cond = ifOp.getCondition();
    if (cond.getType().isInteger(1)) {
      cond = builder
                 .create<UnrealizedConversionCastOp>(
                     ifOp.getLoc(), TypeRange{smt::BoolType::get(ctx)},
                     ValueRange{cond})
                 .getResult(0);
    }

    auto iteOp =
        builder.create<smt::IteOp>(ifOp.getLoc(), cond, *thenValue, *elseValue);
    ifOp.getResult(0).replaceAllUsesWith(iteOp.getResult());
    ifOp.erase();
  }

  std::string loweredName = ("smt_" + structDef.getSymName()).str();
  auto smtFuncRef = llzk::replaceLeaf(
      structRef, StringAttr::get(structDef.getContext(), loweredName));

  auto loweredFunc = cloned.lookupSymbol<func::FuncOp>(smtFuncRef);
  if (!loweredFunc) {
    cloned.emitError() << "could not find lowered SMT function " << loweredName;
    return failure();
  }

  return loweredFunc;
}

namespace lleq {

LogicalResult emitSMTLIBEncoding(component::StructDefOp structDef,
                                 llvm::raw_ostream &os,
                                 llvm::StringRef fieldName) {
  FailureOr<func::FuncOp> loweredFunc = lowerToSMT(structDef, fieldName);
  if (failed(loweredFunc)) {
    return failure();
  }

  SMTLIBFunctionEmitter emitter(os);
  return emitter.emit(*loweredFunc);
}

} // namespace lleq
