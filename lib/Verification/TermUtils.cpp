/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/TermUtils.h"
#include "Analysis/ScalarSymbolAnalysis.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/Array/IR/Types.h>
#include <llzk/Dialect/Bool/IR/Enums.h>
#include <llzk/Dialect/Bool/IR/Ops.h>
#include <llzk/Dialect/Cast/IR/Ops.h>
#include <llzk/Dialect/Constrain/IR/Ops.h>
#include <llzk/Dialect/Felt/IR/Ops.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Util/DynamicAPIntHelper.h>
#include <llzk/Util/TypeHelper.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>

using namespace llzk;
using namespace mlir;

using array::ReadArrayOp;
using array::WriteArrayOp;
using component::MemberReadOp;
using component::MemberWriteOp;
using constrain::EmitEqualityOp;
using felt::FeltConstantOp;

namespace lleq {

void TermBuilder::populateSubcomponent(llzk::component::StructDefOp subcmp) {
  std::string subcmpName = subcmp.getSymName().str();
  // Materialize a sort: `struct.def @B` => `(decl-sort B)`
  cvc5::Sort sort = mgr.mkUninterpretedSort(subcmpName);
  subcmpSorts.insert({subcmp.getType(), sort});
  // `@B::@product(...) -> <@B>` => `(decl-fun create-B (...) B)`
  std::vector<cvc5::Sort> argumentSorts;
  for (auto type : subcmp.getComputeFuncOp().getFunctionType().getInputs()) {
    argumentSorts.push_back(_sort_of_type(type));
  }
  cvc5::Sort initFuncSort = mgr.mkFunctionSort(std::move(argumentSorts), sort);
  cvc5::Term initFunc = mgr.mkConst(initFuncSort, "init-" + subcmpName);
  subcmpInits.insert({subcmp.getType(), initFunc});

  // For each `@B::struct.member @foo : T` => `(decl-fun read-B-foo (B) T)`
  for (auto memberDef : subcmp.getMemberDefs()) {
    cvc5::Sort memberReadFuncSort =
        mgr.mkFunctionSort({sort}, _sort_of_type(memberDef.getType()));
    cvc5::Term memberReadFunc =
        mgr.mkConst(memberReadFuncSort,
                    "read-" + subcmpName + "-" + memberDef.getSymName().str());
    subcmpMembers.insert({memberDef, memberReadFunc});
  }
}

static inline void traverse(cvc5::Term term, const TermBuilder::TermSet &vars,
                            TermBuilder::TermSet &acc) {
  if (vars.find(term) != vars.end()) {
    acc.insert(term);
  } else {
    for (auto child : term) {
      traverse(child, vars, acc);
    }
  }
}

TermBuilder::TermSet TermBuilder::getExtraDecls(cvc5::Term term) {
  TermSet vars;
  for (const auto &[_, c] : constants) {
    vars.insert(c);
  }
  for (const auto &[_, c] : witnessMembers) {
    vars.insert(c);
  }
  for (const auto &[_, c] : constraintMembers) {
    vars.insert(c);
  }

  TermSet decls;
  traverse(term, vars, decls);
  return decls;
}

// Returns the array length for an ArrayType, or nullopt
static inline std::optional<int64_t> sizeOfType(Type type) {
  if (auto arrType = dyn_cast<llzk::array::ArrayType>(type)) {
    auto shape = arrType.getShape();
    llzk::ensure(shape.size() == 1, "multidimensional arrays not supported");
    return shape.front();
  }
  return {};
}

TermBuilder::TermSet TermBuilder::getDeclBounds(TermSet decls,
                                                llvm::DynamicAPInt prime) {
  TermSet bounds;
  for (auto var : decls) {
    if (var.getSort().isArray()) {
      if (var.getSort().getArrayElementSort().isUninterpretedSort()) {
        // Don't need bounds for subcomponent arrays
        continue;
      }
      std::optional<size_t> size;
      if (auto it = termTypes.find(var); it != termTypes.end()) {
        size = sizeOfType(it->second);
      }
      auto bound = _array_quantified_term(
          [this, &var, &prime](cvc5::Term index) {
            return _is_mod(_array_read_impl(var, index), prime);
          },
          size);
      bounds.insert(bound);
    } else if (!var.getSort().isUninterpretedSort()) {
      // Don't need bounds for subcomponents
      bounds.insert(_is_mod(var, prime));
    }
  }
  return bounds;
}

static inline cvc5::Sort getElementSortOrSelf(cvc5::Sort s) {
  if (s.isArray()) {
    return getElementSortOrSelf(s.getArrayElementSort());
  }
  return s;
}

void TermBuilder::emitSubcmpDeclarations(llvm::raw_ostream &os) {
  if (subcmpSorts.empty()) {
    return;
  }
  os << "; Subcomponents\n";
  for (const auto &[subcmp, sort] : subcmpSorts) {
    os << "(declare-sort " << sort.toString() << " 0)\n";
    auto it = subcmpInits.find(subcmp);
    llzk::ensure(it != subcmpInits.end(), "unknown subcomponent type");
    auto initFunc = it->second;
    auto argTypes = initFunc.getSort().getFunctionDomainSorts();
    os << "(declare-fun " << initFunc.toString() << " (";
    llvm::interleave(
        argTypes, os, [&os](cvc5::Sort sort) { os << sort.toString(); }, " ");
    os << ") " << sort.toString() << ")\n";
  }

  for (const auto &[_, memberRead] : subcmpMembers) {
    auto fnSort = memberRead.getSort();
    os << "(declare-fun " << memberRead.toString() << " ("
       << fnSort.getFunctionDomainSorts().front().toString() << ") "
       << fnSort.getFunctionCodomainSort().toString() << ")\n";
  }
}

cvc5::Sort TermBuilder::_sort_of_type(Type type) {
  // TODO: subcomponent
  if (auto structType = dyn_cast<llzk::component::StructType>(type)) {
    auto it = subcmpSorts.find(structType);
    llzk::ensure(it != subcmpSorts.end(), "unknown subcomponent type");
    return it->second;
  }
  if (type.isSignlessInteger() &&
      dyn_cast<IntegerType>(type).getIntOrFloatBitWidth() == 1) {
    return mgr.getBooleanSort();
  }
  if (auto arrType = dyn_cast<llzk::array::ArrayType>(type)) {
    return mgr.mkArraySort(mgr.getIntegerSort(),
                           _sort_of_type(arrType.getElementType()));
  }
  return mgr.getIntegerSort();
}

cvc5::Term TermBuilder::getExpression(mlir::Value value) {
  // If we've already cached a value, just look it up
  if (auto it = expressions.find(value); it != expressions.end()) {
    return it->second;
  }

  Operation *op = value.getDefiningOp();
  // TODO: we could theoretically handle a value yielded from an scf.if with
  // multiple values
  if (op == nullptr || op->getNumResults() != 1) {
    return getConstant(value);
  }

  // If its a basic arithmetic operation we can build it directly
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
    auto expression =
        mgr.mkTerm(it->second, {operandTerms.begin(), operandTerms.end()});
    expressions.insert({value, expression});
    return expression;
  }

  // Otherwise, switch on the type of the operation
  auto expression =
      llvm::TypeSwitch<Operation *, cvc5::Term>(op)
          .Case<boolean::CmpOp>([this, op](boolean::CmpOp cmp) {
            static llvm::DenseMap<boolean::FeltCmpPredicate, cvc5::Kind>
                predicateToKind = {
                    {boolean::FeltCmpPredicate::EQ, cvc5::Kind::EQUAL},
                    {boolean::FeltCmpPredicate::LT, cvc5::Kind::LT},
                    {boolean::FeltCmpPredicate::LE, cvc5::Kind::LEQ},
                    {boolean::FeltCmpPredicate::GT, cvc5::Kind::GT},
                    {boolean::FeltCmpPredicate::GE, cvc5::Kind::GEQ}};
            SmallVector<cvc5::Term> operandTerms{
                llvm::map_to_vector(op->getOperands(), [this](Value value) {
                  return getExpression(value);
                })};
            return mgr.mkTerm(predicateToKind.at(cmp.getPredicate()),
                              {operandTerms.begin(), operandTerms.end()});
          })
          .Case<MemberReadOp>([this](MemberReadOp read) {
            read.getType();
            return getConstant(read.getMemberName(), read.getType(),
                               isWitnessOp(read));
          })
          .Case<ReadArrayOp>([this](ReadArrayOp read) {
            llzk::ensure(read.getIndices().size() == 1,
                         "multidimensional arrays are not supported");
            return arrayRead(read.getArrRef(), read.getIndices().front());
          })
          .Case<FeltConstantOp>([this](FeltConstantOp constOp) {
            SmallString<64> str;
            constOp.getValue().getValue().toStringUnsigned(str);
            return mgr.mkInteger(std::string{str});
          })
          .Case<arith::ConstantIntOp, arith::ConstantIndexOp>(
              [this](auto constOp) {
                auto val = dyn_cast<IntegerAttr>(constOp.getValue()).getValue();
                return getInteger(val);
              })
          .Case<array::CreateArrayOp>([this](array::CreateArrayOp createArr) {
            // If the array is written to exactly one struct member later, just
            // materialize a symbol for that directly
            auto destination = getArrayDestination(createArr.getResult());
            if (destination.has_value()) {
              return getConstant(destination->getMemberName(),
                                 createArr.getType(), true);
            }
            return getConstant(createArr.getResult());
          })
          .Case<cast::FeltToIndexOp, cast::IntToFeltOp>(
              [this](auto op) { return getExpression(op.getValue()); })
          .Case<UnrealizedConversionCastOp>(
              [this](UnrealizedConversionCastOp cast) {
                return getExpression(cast.getInputs().front());
              })
          .Case<scf::IfOp>([this](scf::IfOp ifOp) {
            auto trueValue =
                getExpression(ifOp.thenYield().getResults().front());
            auto falseValue =
                getExpression(ifOp.elseYield().getResults().front());
            auto condition = getExpression(ifOp.getCondition());
            return mgr.mkTerm(cvc5::Kind::ITE,
                              {condition, trueValue, falseValue});
          })
          .Case<llzk::function::CallOp>([this](llzk::function::CallOp call) {
            // For now just deal with calls to @compute and error out on other
            // function calls
            SymbolTableCollection tables;
            llzk::ensure(call.calleeIsCompute(),
                         "arbitrary function calls not supported yet");
            auto target = call.getCalleeTarget(tables);
            llzk::ensure(succeeded(target), "failed to resolve callee target");
            SmallVector<Value> args = call.getArgOperands();
            return initSubcmp(
                target->get()->getParentOfType<component::StructDefOp>(), args);
          })
          .Default([op](auto) -> cvc5::Term {
            llvm::report_fatal_error("unknown op: " +
                                     op->getName().getStringRef());
          });

  expressions.insert({value, expression});
  return expression;
}

cvc5::Term TermBuilder::getConstant(Value value) {
  if (auto it = constants.find(value); it != constants.end()) {
    return it->second;
  }

  std::optional<std::string> name;
  // if (auto blockArg = dyn_cast<BlockArgument>(value)) {
  //   name.emplace("arg" + std::to_string(blockArg.getArgNumber()));
  // }

  auto newConst = mgr.mkConst(_sort_of_type(value.getType()), name);
  constants.insert({value, newConst});
  termTypes.insert({newConst, value.getType()});
  return newConst;
}

cvc5::Term TermBuilder::getConstant(llzk::component::MemberDefOp memberDef,
                                    bool isWitness) {
  return getConstant(memberDef.getSymName(), memberDef.getType(), isWitness);
}
cvc5::Term TermBuilder::getConstant(StringRef memberName, Type type,
                                    bool isWitness) {
  auto &memberMap = isWitness ? witnessMembers : constraintMembers;

  if (auto it = memberMap.find(memberName); it != memberMap.end()) {
    return it->second;
  }

  auto newTerm = mgr.mkConst(_sort_of_type(type),
                             (memberName + (isWitness ? "_w" : "_c")).str());
  memberMap.insert({memberName, newTerm});
  termTypes.insert({newTerm, type});
  return newTerm;
}

cvc5::Term TermBuilder::getInteger(llvm::DynamicAPInt val) {
  std::string str;
  llvm::raw_string_ostream os{str};
  val.print(os);
  return mgr.mkInteger(str);
}

cvc5::Term TermBuilder::initSubcmp(llzk::component::StructDefOp subcmp,
                                   llvm::ArrayRef<Value> args) {
  auto it = subcmpInits.find(subcmp.getType());
  llzk::ensure(it != subcmpInits.end(),
               "unknown subcomponent: " + subcmp.getSymName().str());

  std::vector<cvc5::Term> termArgs{it->second};
  termArgs.reserve(args.size() + 1);

  for (auto arg : args) {
    termArgs.push_back(getConstant(arg));
  }
  return mgr.mkTerm(cvc5::Kind::APPLY_UF, termArgs);
}

cvc5::Term TermBuilder::readSubcmpMember(mlir::Value subcmp,
                                         llzk::component::MemberDefOp member) {
  auto it = subcmpMembers.find(member);
  llzk::ensure(it != subcmpMembers.end(),
               "unknown subcomponent member: " + member.getSymName().str());

  return mgr.mkTerm(cvc5::Kind::APPLY_UF, {it->second, getConstant(subcmp)});
}

cvc5::Term TermBuilder::_is_mod(cvc5::Term val, llvm::DynamicAPInt mod) {
  llzk::ensure(val.getSort().isInteger(),
               "cannot bound non-integral sort modulo");
  return mgr.mkTerm(cvc5::Kind::LEQ, {getInteger(0), val, getInteger(mod - 1)});
}

cvc5::Term TermBuilder::_reduce_mod_impl(cvc5::Term val,
                                         llvm::DynamicAPInt mod) {
  return mgr.mkTerm(cvc5::Kind::INTS_MODULUS, {val, getInteger(mod)});
}

// Returns the array length if either term is an array of known length, nullopt
// if neither, and asserts failure if the lengths differ
static inline std::optional<int64_t> getArraySize(
    cvc5::Term a, cvc5::Term b,
    std::unordered_map<cvc5::Term, Type, std::hash<cvc5::Term>> termTypes) {
  std::optional<int64_t> arraySize;
  if (auto ait = termTypes.find(a); ait != termTypes.end()) {
    arraySize = sizeOfType(ait->second);
  }
  if (auto bit = termTypes.find(b); bit != termTypes.end()) {
    auto size = sizeOfType(bit->second);
    llzk::ensure(!arraySize.has_value() || arraySize == size,
                 "incompatible array sizes");
    return size;
  }
  return arraySize;
}

cvc5::Term TermBuilder::_array_quantified_term(
    std::function<cvc5::Term(cvc5::Term)> builder,
    std::optional<int64_t> size) {

  auto index = mgr.mkVar(mgr.getIntegerSort(), "i");
  auto forallBody = builder(index);
  if (size.has_value()) {
    forallBody =
        mgr.mkTerm(cvc5::Kind::IMPLIES,
                   {_is_mod(index, llzk::toDynamicAPInt(*size)), forallBody});
  }

  return mgr.mkTerm(
      cvc5::Kind::FORALL,
      {mgr.mkTerm(cvc5::Kind::VARIABLE_LIST, {index}), forallBody});
}

cvc5::Term TermBuilder::_assert_equal_impl(cvc5::Term a, cvc5::Term b) {
  Value arrVal;
  if (a.getSort().isArray() && b.getSort().isArray()) {
    auto size = getArraySize(a, b, termTypes);
    return _array_quantified_term(
        [this, &a, &b](cvc5::Term index) {
          return _assert_equal_impl(_array_read_impl(a, index),
                                    _array_read_impl(b, index));
        },
        size);
  }

  if (a.getSort().isUninterpretedSort() && b.getSort().isUninterpretedSort()) {
    return mgr.mkTerm(cvc5::Kind::EQUAL, {a, b});
  }

  return mgr.mkTerm(cvc5::Kind::EQUAL, {_reduce_mod_impl(a, field.prime()),
                                        _reduce_mod_impl(b, field.prime())});
}

cvc5::Term TermBuilder::_array_read_impl(cvc5::Term arr, cvc5::Term index) {
  return mgr.mkTerm(cvc5::Kind::SELECT, {arr, index});
}

cvc5::Term TermBuilder::_array_write_impl(cvc5::Term arr, cvc5::Term index,
                                          cvc5::Term elem) {
  return mgr.mkTerm(cvc5::Kind::STORE, {arr, index, elem});
}

cvc5::Term ImplicationTerm::buildTerm(cvc5::TermManager &mgr) {
  if (antecedents.empty()) {
    return consequent;
  }

  auto antecedent = antecedents.size() == 1
                        ? antecedents.front()
                        : mgr.mkTerm(cvc5::Kind::AND,
                                     {antecedents.begin(), antecedents.end()});
  return mgr.mkTerm(cvc5::Kind::IMPLIES, {antecedent, consequent});
}

void ImplicationTerm::substitute(cvc5::Term oldTerm, cvc5::Term newTerm) {
  for (auto &antecedent : antecedents) {
    antecedent = antecedent.substitute(oldTerm, newTerm);
  }
  consequent = consequent.substitute(oldTerm, newTerm);
}

void ConjunctionTerm::addConjunct(const ImplicationTerm &term) {
  conjuncts.push_back(term);
}

void ConjunctionTerm::substitute(cvc5::Term oldTerm, cvc5::Term newTerm) {
  for (auto &conjunct : conjuncts) {
    conjunct.substitute(oldTerm, newTerm);
  }
}

cvc5::Term ConjunctionTerm::buildTerm(cvc5::TermManager &mgr) {
  llzk::ensure(!conjuncts.empty(), "cannot build term from empty conjunction");

  if (conjuncts.size() == 1) {
    return conjuncts.front().buildTerm(mgr);
  }
  auto builtConjuncts = llvm::map_to_vector(
      conjuncts, [&mgr](ImplicationTerm term) { return term.buildTerm(mgr); });
  return mgr.mkTerm(cvc5::Kind::AND,
                    {builtConjuncts.begin(), builtConjuncts.end()});
}

void ConjunctionTerm::addAntecedent(cvc5::Term term) {
  for (auto &conjunct : conjuncts) {
    conjunct.addAntecedent(term);
  }
}

} // namespace lleq
