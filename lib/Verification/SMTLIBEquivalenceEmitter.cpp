/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/SMTLIBEquivalenceEmitter.h"

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LLVM.h>
#include <optional>
#include <string>
#include <utility>

#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/SMT/IR/SMTOps.h>
#include <llzk/Dialect/Struct/IR/Ops.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace mlir;
using namespace llzk;

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
    auto diag = func.emitError();
    for (auto [index, arg] : llvm::enumerate(func.getArguments())) {
      std::string name = "arg" + std::to_string(index);
      values[arg] = name;
      os << "(declare-const " << name << " " << sortForType(arg.getType(), diag)
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

    auto diag = op->emitError();
    os << "(declare-fun " << symbol << " " << sortForType(op.getType(), diag)
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

  // LogicalResult processBlockWithState(Block &block,
  //                                     llvm::DenseMap<Value, std::string>
  //                                     &state, const
  //                                     std::optional<std::string> &guard) {
  //   auto oldValues = std::move(values);
  //   values = std::move(state);
  //   LogicalResult result = processBlock(block, guard);
  //   state = std::move(values);
  //   values = std::move(oldValues);
  //   return result;
  // }

  FailureOr<std::string> buildExpression(Operation *op) {

    static DenseMap<StringRef, StringRef> opToSmtOp = {
        {"smt.int.neg", "-"},      {"smt.not", "not"},
        {"smt.int.add", "+"},      {"smt.int.mul", "*"},
        {"smt.int.sub", "-"},      {"smt.int.mod", "mod"},
        {"smt.int.eq", "="},       {"smt.int.and", "and"},
        {"smt.int.or", "or"},      {"smt.int.xor", "xor"},
        {"smt.int.implies", "=>"}, {"smt.int.not", "not"},
        {"smt.int.ite", "ite"},
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

  FailureOr<std::string> sortForType(Type type, InFlightDiagnostic &diag) {
    return TypeSwitch<Type, FailureOr<std::string>>(type)
        .Case<smt::IntType>([](auto) { return success("Int"); })
        .Case<smt::BoolType>([](auto) { return success("Bool"); })
        .Default([&diag](Type type) -> FailureOr<std::string> {
          if (type.isInteger(1)) {
            return success("Bool");
          }
          diag << "unsupported SMT sort: " << type;
          return failure();
        });
  }

  llvm::raw_ostream &os;
  llvm::DenseMap<Value, std::string> values;

  unsigned nextTempId = 0;
};

#if LLEQ_HAS_SMT_BACKEND

// LogicalResult lowerStructToSMT(ModuleOp module, llvm::StringRef
// rootStructName,
//                                const std::optional<std::string> &fieldName) {
//   llvm::StringRef normalizedRoot = stripSigil(rootStructName);
//   auto rootStruct =
//       module.lookupSymbol<llzk::component::StructDefOp>(normalizedRoot);
//   if (!rootStruct) {
//     return module.emitError()
//            << "could not find root struct @" << normalizedRoot;
//   }

//   if (failed(ensureProductFunc(module, rootStruct))) {
//     return failure();
//   }

//   PassManager pm(module.getContext(), PassManager::getAnyOpAnchorName(),
//                  PassManager::Nesting::Implicit);
//   pm.enableVerifier(false);

//   auto smtPass = llzk::smt::createSMTLoweringPass();
//   if (fieldName) {
//     std::string options = "field=" + *fieldName;
//     if (failed(smtPass->initializeOptions(options, [&](const llvm::Twine
//     &err) {
//           return module.emitError() << err;
//         }))) {
//       return failure();
//     }
//   }
//   pm.addPass(std::move(smtPass));
//   pm.addPass(createCanonicalizerPass());
//   pm.addPass(createCSEPass());

//   return pm.run(module);
// }
#endif

namespace lleq {

// LogicalResult
// emitSMTLIBEquivalence(mlir::ModuleOp module, llvm::raw_ostream &os,
//                       llvm::StringRef memberName,
//                       llvm::StringRef rootStructName,
//                       const std::optional<std::string> &fieldName) {
//   llvm::StringRef normalizedRoot = stripSigil(rootStructName);
//   auto rootStruct =
//       module.lookupSymbol<llzk::component::StructDefOp>(normalizedRoot);
//   if (!rootStruct) {
//     return module.emitError()
//            << "could not find root struct @" << normalizedRoot;
//   }

//   llvm::StringRef normalizedMember = stripSigil(memberName);
//   if (!rootStruct.getMemberDef(
//           StringAttr::get(module.getContext(), normalizedMember))) {
//     return rootStruct.emitError()
//            << "could not find member @" << normalizedMember << " in struct @"
//            << normalizedRoot;
//   }

// #if !LLEQ_HAS_SMT_BACKEND
//   (void)fieldName;
//   return module.emitError()
//          << "SMT equivalence emission is unavailable in this build because
//          the "
//             "active LLZK package does not expose the SMT lowering backend";
// #else
//   auto cloned = cast<ModuleOp>(module->clone());
//   if (failed(lowerStructToSMT(cloned, normalizedRoot, fieldName))) {
//     return failure();
//   }

//   std::string loweredName = ("smt_" + normalizedRoot).str();
//   auto loweredFunc = cloned.lookupSymbol<func::FuncOp>(loweredName);
//   if (!loweredFunc) {
//     return cloned.emitError()
//            << "could not find lowered SMT function @" << loweredName;
//   }

//   SMTLIBFunctionEmitter emitter(os);
//   return emitter.emit(loweredFunc, normalizedMember);
// #endif
// }

} // namespace lleq
