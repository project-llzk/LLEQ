/**
 * Copyright 2026 Project LLZK
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/TermUtils.h"
#include "Analysis/ScalarSymbolAnalysis.h"
#include "Verification/Utils.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/Array/IR/Types.h>
#include <llzk/Dialect/Bool/IR/Enums.h>
#include <llzk/Dialect/Bool/IR/Ops.h>
#include <llzk/Dialect/Cast/IR/Ops.h>
#include <llzk/Dialect/Constrain/IR/Ops.h>
#include <llzk/Dialect/Felt/IR/Ops.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/LLZK/IR/Ops.h>
#include <llzk/Dialect/Polymorphic/IR/Ops.h>
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
using component::MemberReadOp;
using felt::FeltConstantOp;

// Replace !struct.type<@S<[]>> with !struct.type<@S>
static component::StructType normalize(component::StructType type) {
  if (type.getParams() && type.getParams().empty()) {
    return component::StructType::get(type.getNameRef());
  }
  return type;
}

namespace lleq {

// Materialize a sort: `struct.def @B` => `(decl-sort B)`
void TermBuilder::registerSubcomponentSort(component::StructDefOp subcmpDef) {
  auto name = subcmpDef.getSymName().str();
  auto sort = mgr.mkUninterpretedSort(name);
  subcmpSorts.insert({normalize(subcmpDef.getType()), sort});
}

void TermBuilder::registerSubcomponentFuncs(component::StructDefOp subcmpDef) {
  util::ensureProductFunc(subcmpDef->getParentOfType<ModuleOp>(), subcmpDef);
  auto subcmpType = normalize(subcmpDef.getType());
  auto name = subcmpDef.getSymName().str();
  auto it = subcmpSorts.find(subcmpType);
  ensure(it != subcmpSorts.end(),
         "Cannot register funcs for unregistered subcomponent: " +
             subcmpDef.getSymName());
  auto sort = it->second;

  // `@B::@product(...) -> <@B>` => `(decl-fun create-B (...) B)`
  std::vector<cvc5::Sort> argumentSorts;
  for (auto type : subcmpDef.getProductFuncOp().getFunctionType().getInputs()) {
    argumentSorts.push_back(_sort_of_type(type));
  }

  auto initSort = argumentSorts.empty()
                      ? sort
                      : mgr.mkFunctionSort(std::move(argumentSorts), sort);
  subcmpInits.insert({subcmpType, mgr.mkConst(initSort, "init-" + name)});

  // For each `@B::struct.member @foo : T` => `(decl-fun read-B-foo (B) T)`
  for (auto memberDef : subcmpDef.getMemberDefs()) {
    auto memberReadFuncSort =
        mgr.mkFunctionSort({sort}, _sort_of_type(memberDef.getType()));
    cvc5::Term memberReadFunc =
        mgr.mkConst(memberReadFuncSort,
                    "read-" + name + "-" + memberDef.getSymName().str());
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

// Returns the full shape for an ArrayType, or nullopt for non-array types.
static inline std::optional<ArrayShape> shapeOfType(Type type) {
  if (auto arrType = dyn_cast<array::ArrayType>(type)) {
    return ArrayShape(arrType.getShape().begin(), arrType.getShape().end());
  }
  return {};
}

TermBuilder::TermSet TermBuilder::getDeclBounds(TermSet decls,
                                                llvm::DynamicAPInt prime) {
  TermSet bounds = auxiliaryBounds;
  for (auto var : decls) {
    if (var.getSort().isArray()) {
      cvc5::Sort elementSort = var.getSort();
      while (elementSort.isArray()) {
        elementSort = elementSort.getArrayElementSort();
      }
      if (elementSort.isUninterpretedSort()) {
        // Don't need bounds for subcomponent arrays
        continue;
      }
      ArrayShape shape;
      if (auto it = termTypes.find(var); it != termTypes.end()) {
        if (auto varShape = shapeOfType(it->second)) {
          shape = *varShape;
        }
      }
      auto bound = _array_quantified_term(
          [this, &var, &prime](ArrayRef<cvc5::Term> indices) {
            return _is_mod(_array_read_impl(var, indices), prime);
          },
          shape);
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
    auto it = subcmpInits.find(normalize(subcmp));
    ensure(it != subcmpInits.end(), "unknown subcomponent type");
    auto initFunc = it->second;
    std::vector<cvc5::Sort> argTypes;
    if (auto initSort = initFunc.getSort(); initSort.isFunction()) {
      argTypes = initSort.getFunctionDomainSorts();
    }
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
  if (auto structType = dyn_cast<component::StructType>(type)) {
    auto it = subcmpSorts.find(normalize(structType));
    ensure(it != subcmpSorts.end(), "unknown subcomponent type");
    return it->second;
  }
  if (type.isSignlessInteger(1)) {
    return mgr.getBooleanSort();
  }
  if (auto arrType = dyn_cast<array::ArrayType>(type)) {
    cvc5::Sort sort = _sort_of_type(arrType.getElementType());
    // LLZK stores all extents on one ArrayType, but SMT arrays are rank-1, so
    // multidimensional arrays must be materialized as nested array sorts.
    for (int64_t extent : llvm::reverse(arrType.getShape())) {
      (void)extent;
      sort = mgr.mkArraySort(mgr.getIntegerSort(), sort);
    }
    return sort;
  }
  return mgr.getIntegerSort();
}

template <class PredicateT>
static llvm::DenseMap<PredicateT, cvc5::Kind> predicateToKind;

template <>
llvm::DenseMap<boolean::FeltCmpPredicate, cvc5::Kind>
    predicateToKind<boolean::FeltCmpPredicate> = {
        {boolean::FeltCmpPredicate::EQ, cvc5::Kind::EQUAL},
        {boolean::FeltCmpPredicate::LT, cvc5::Kind::LT},
        {boolean::FeltCmpPredicate::LE, cvc5::Kind::LEQ},
        {boolean::FeltCmpPredicate::GT, cvc5::Kind::GT},
        {boolean::FeltCmpPredicate::GE, cvc5::Kind::GEQ}};

template <>
llvm::DenseMap<arith::CmpIPredicate, cvc5::Kind>
    predicateToKind<arith::CmpIPredicate> = {
        {arith::CmpIPredicate::eq, cvc5::Kind::EQUAL},
        {arith::CmpIPredicate::slt, cvc5::Kind::LT},
        {arith::CmpIPredicate::sle, cvc5::Kind::LEQ},
        {arith::CmpIPredicate::sgt, cvc5::Kind::GT},
        {arith::CmpIPredicate::sge, cvc5::Kind::GEQ},
        {arith::CmpIPredicate::ult, cvc5::Kind::LT},
        {arith::CmpIPredicate::ule, cvc5::Kind::LEQ},
        {arith::CmpIPredicate::ugt, cvc5::Kind::GT},
        {arith::CmpIPredicate::uge, cvc5::Kind::GEQ}};

template <class PredicateT> PredicateT eqPred;
template <class PredicateT> PredicateT neqPred;

template <>
auto eqPred<boolean::FeltCmpPredicate> = boolean::FeltCmpPredicate::EQ;

template <>
auto neqPred<boolean::FeltCmpPredicate> = boolean::FeltCmpPredicate::NE;

template <> auto eqPred<arith::CmpIPredicate> = arith::CmpIPredicate::eq;

template <> auto neqPred<arith::CmpIPredicate> = arith::CmpIPredicate::ne;

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
      {felt::AddFeltOp::getOperationName(), cvc5::Kind::ADD},
      {felt::SubFeltOp::getOperationName(), cvc5::Kind::SUB},
      {felt::MulFeltOp::getOperationName(), cvc5::Kind::MULT},
      {felt::SignedModFeltOp::getOperationName(), cvc5::Kind::INTS_MODULUS},
      {felt::UnsignedModFeltOp::getOperationName(), cvc5::Kind::INTS_MODULUS},
      {felt::SignedIntDivFeltOp::getOperationName(), cvc5::Kind::INTS_DIVISION},
      {felt::UnsignedIntDivFeltOp::getOperationName(),
       cvc5::Kind::INTS_DIVISION},
      {felt::NegFeltOp::getOperationName(), cvc5::Kind::NEG},
      {arith::AddIOp::getOperationName(), cvc5::Kind::ADD},
      {arith::SubIOp::getOperationName(), cvc5::Kind::SUB},
      {boolean::OrBoolOp::getOperationName(), cvc5::Kind::OR},
      {boolean::AndBoolOp::getOperationName(), cvc5::Kind::AND}};

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
          .Case<felt::DivFeltOp>(
              // Make sure we perform division in the field, not truncating
              // integer division
              [this](felt::DivFeltOp divOp) {
                // Build a new constant for the division result and constrain it
                auto result = getConstant(divOp.getResult());
                // assert result * denominator == numerator if denominator is
                // nonzero
                auto denominator = getExpression(divOp.getRhs());
                auto nonzeroBound = assertEqual(
                    mgr.mkTerm(cvc5::Kind::MULT, {result, denominator}),
                    getExpression(divOp.getLhs()));
                auto zeroBound = assertEqual(result, getInteger(0));

                auxiliaryBounds.insert(mgr.mkTerm(
                    cvc5::Kind::ITE, {assertEqual(denominator, getInteger(0)),
                                      zeroBound, nonzeroBound}));

                return result;
              })
          .Case<polymorphic::ConstReadOp>(
              [this](polymorphic::ConstReadOp constRead) {
                return getConstant(constRead.getConstName());
              })
          .Case<boolean::CmpOp, arith::CmpIOp>([this, op](auto cmp) {
            SmallVector<cvc5::Term> operandTerms{
                llvm::map_to_vector(op->getOperands(), [this](Value value) {
                  return getExpression(value);
                })};

            using PredT = decltype(cmp.getPredicate());

            if (cmp.getPredicate() == neqPred<PredT>) {
              return mgr
                  .mkTerm(predicateToKind<PredT>.at(eqPred<PredT>),
                          {operandTerms.begin(), operandTerms.end()})
                  .notTerm();
            }
            return mgr.mkTerm(predicateToKind<PredT>.at(cmp.getPredicate()),
                              {operandTerms.begin(), operandTerms.end()});
          })
          .Case<MemberReadOp>([this](MemberReadOp read) {
            if (read.getComponent().getType() ==
                read->getParentOfType<component::StructDefOp>().getType()) {
              // Not a subcomponent read, so just return the member constant
              return getConstant(read.getMemberName(), read.getType(),
                                 isWitnessOp(read));
            }
            SymbolTableCollection tables;
            auto memberDef = read.getMemberDefOp(tables)->get();
            return readSubcmpMember(read.getComponent(), memberDef);
            // return subcmpMembers.at(memberDef);
          })
          .Case<ReadArrayOp>([this](ReadArrayOp read) {
            SmallVector<Value> indices(read.getIndices().begin(),
                                       read.getIndices().end());
            return arrayRead(read.getArrRef(), indices);
          })
          .Case<FeltConstantOp>([this](FeltConstantOp constOp) {
            SmallString<64> str;
            constOp.getValue().getValue().toStringSigned(str);
            return mgr.mkInteger(std::string{str});
          })
          .Case<arith::ConstantIntOp, arith::ConstantIndexOp>(
              [this](auto constOp) {
                auto val = dyn_cast<IntegerAttr>(constOp.getValue()).getValue();
                Type constType = constOp.getType();
                return getInteger(val, constType.isInteger(1));
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
          .Case<NonDetOp>([this](NonDetOp nondet) {
            if (isa<array::ArrayType>(nondet.getType())) {
              // If its using llzk.nondet to initialize an array, just copy the
              // array logic
              auto destination = getArrayDestination(nondet.getResult());
              if (destination.has_value()) {
                return getConstant(destination->getMemberName(),
                                   nondet.getType(), true);
              }
            }

            return getConstant(nondet.getResult());
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
          .Case<scf::ForOp>([this](auto) -> cvc5::Term {
            llvm::report_fatal_error("loop-yielded values not yet supported");
          })
          .Case<function::CallOp>([this](function::CallOp call) {
            // For now just deal with calls to @compute and error out on other
            // function calls
            SymbolTableCollection tables;
            ensure(call.calleeIsStructCompute() || call.calleeIsStructProduct(),
                   "arbitrary function calls not supported yet");
            auto target = call.getCalleeTarget(tables);
            ensure(succeeded(target), "failed to resolve callee target");
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
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (auto funcDef = dyn_cast<function::FuncDefOp>(
            blockArg.getParentBlock()->getParentOp())) {
      auto argName = funcDef.getArgNameAttr(blockArg.getArgNumber());
      if (argName.has_value()) {
        name.emplace(argName->getValue());
      } else {
        name.emplace((funcDef.getSymName() + "_" +
                      std::to_string(blockArg.getArgNumber()))
                         .str());
      }
    }
  }

  auto newConst = mgr.mkConst(_sort_of_type(value.getType()), name);
  constants.insert({value, newConst});
  termTypes.insert({newConst, value.getType()});
  return newConst;
}

void TermBuilder::addEquivalentMember(component::MemberDefOp memberDef) {
  StringRef memberName = memberDef.getSymName();
  auto memberConst =
      mgr.mkConst(_sort_of_type(memberDef.getType()), memberName.str());

  witnessMembers.insert({memberName, memberConst});
  constraintMembers.insert({memberName, memberConst});
  termTypes.insert({memberConst, memberDef.getType()});
}

cvc5::Term TermBuilder::getConstant(component::MemberDefOp memberDef,
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

cvc5::Term TermBuilder::getConstant(StringRef symbolName) {
  if (auto it = polyMembers.find(symbolName); it != polyMembers.end()) {
    return it->second;
  }
  auto newTerm = mgr.mkConst(mgr.getIntegerSort(), std::string{symbolName});
  polyMembers.insert({symbolName, newTerm});
  return newTerm;
}

cvc5::Term TermBuilder::getInteger(llvm::DynamicAPInt val, bool asBoolean) {
  if (asBoolean) {
    return mgr.mkBoolean(val != 0);
  }
  std::string str;
  llvm::raw_string_ostream os{str};
  val.print(os);
  return mgr.mkInteger(str);
}

cvc5::Term TermBuilder::initSubcmp(component::StructDefOp subcmp,
                                   llvm::ArrayRef<Value> args) {
  auto it = subcmpInits.find(normalize(subcmp.getType()));
  ensure(it != subcmpInits.end(),
         "unknown subcomponent: " + subcmp.getSymName().str());

  if (args.empty()) {
    // No function to call, just return the symbol directly
    return it->second;
  }

  std::vector<cvc5::Term> termArgs{it->second};
  termArgs.reserve(args.size() + 1);

  for (auto arg : args) {
    termArgs.push_back(getExpression(arg));
  }
  return mgr.mkTerm(cvc5::Kind::APPLY_UF, termArgs);
}

cvc5::Term TermBuilder::readSubcmpMember(mlir::Value subcmp,
                                         component::MemberDefOp member) {
  auto it = subcmpMembers.find(member);
  ensure(it != subcmpMembers.end(),
         "unknown subcomponent member: " + member.getSymName().str());

  return mgr.mkTerm(cvc5::Kind::APPLY_UF, {it->second, getConstant(subcmp)});
}

cvc5::Term TermBuilder::_is_mod(cvc5::Term val, llvm::DynamicAPInt mod) {
  ensure(val.getSort().isInteger(), "cannot bound non-integral sort modulo");
  return mgr.mkTerm(cvc5::Kind::LEQ, {getInteger(0), val, getInteger(mod - 1)});
}

cvc5::Term TermBuilder::_reduce_mod_impl(cvc5::Term val,
                                         llvm::DynamicAPInt mod) {
  return mgr.mkTerm(cvc5::Kind::INTS_MODULUS, {val, getInteger(mod)});
}

// Returns the array shape if either term is an array of known shape, nullopt
// if neither, and asserts failure if the shapes differ.
static inline std::optional<ArrayShape> getArrayShape(
    cvc5::Term a, cvc5::Term b,
    std::unordered_map<cvc5::Term, Type, std::hash<cvc5::Term>> termTypes) {
  std::optional<ArrayShape> arrayShape;
  if (auto ait = termTypes.find(a); ait != termTypes.end()) {
    arrayShape = shapeOfType(ait->second);
  }
  if (auto bit = termTypes.find(b); bit != termTypes.end()) {
    auto shape = shapeOfType(bit->second);
    ensure(!arrayShape.has_value() || arrayShape == shape,
           "incompatible array shapes");
    return shape;
  }
  return arrayShape;
}

cvc5::Term TermBuilder::_array_quantified_term(
    std::function<cvc5::Term(ArrayRef<cvc5::Term>)> builder,
    ArrayRef<int64_t> shape) {

  if (shape.empty()) {
    return builder({});
  }

  std::vector<cvc5::Term> indices;
  indices.reserve(shape.size());
  for (auto [dim, extent] : llvm::enumerate(shape)) {
    (void)extent;
    indices.push_back(
        mgr.mkVar(mgr.getIntegerSort(), "i" + std::to_string(dim)));
  }

  auto forallBody = builder(indices);
  SmallVector<cvc5::Term> bounds;
  for (auto [index, extent] : llvm::zip(indices, shape)) {
    bounds.push_back(_is_mod(index, toDynamicAPInt(extent)));
  }
  if (!bounds.empty()) {
    forallBody =
        mgr.mkTerm(cvc5::Kind::IMPLIES, {conjunctAll(bounds, mgr), forallBody});
  }

  return mgr.mkTerm(
      cvc5::Kind::FORALL,
      {mgr.mkTerm(cvc5::Kind::VARIABLE_LIST, indices), forallBody});
}

cvc5::Term TermBuilder::_assert_equal_impl(cvc5::Term a, cvc5::Term b) {
  Value arrVal;
  if (a.getSort().isArray() && b.getSort().isArray()) {
    auto shape = getArrayShape(a, b, termTypes);
    return _array_quantified_term(
        [this, &a, &b](ArrayRef<cvc5::Term> indices) {
          return _assert_equal_impl(_array_read_impl(a, indices),
                                    _array_read_impl(b, indices));
        },
        shape.value_or(ArrayShape{}));
  }

  if (a.getSort().isUninterpretedSort() && b.getSort().isUninterpretedSort()) {
    return mgr.mkTerm(cvc5::Kind::EQUAL, {a, b});
  }

  return mgr.mkTerm(cvc5::Kind::EQUAL, {_reduce_mod_impl(a, field.prime()),
                                        _reduce_mod_impl(b, field.prime())});
}

cvc5::Term TermBuilder::_array_read_impl(cvc5::Term arr,
                                         ArrayRef<cvc5::Term> indices) {
  ensure(!indices.empty(), "array read requires at least one index");
  ensure(arr.getSort().isArray(), "cannot index into a non-array sort");

  cvc5::Term result = arr;
  for (cvc5::Term index : indices) {
    result = mgr.mkTerm(cvc5::Kind::SELECT, {result, index});
  }
  return result;
}

cvc5::Term TermBuilder::_array_write_impl(cvc5::Term arr,
                                          ArrayRef<cvc5::Term> indices,
                                          cvc5::Term elem) {
  ensure(!indices.empty(), "array write requires at least one index");

  if (indices.size() == 1) {
    return mgr.mkTerm(cvc5::Kind::STORE, {arr, indices.front(), elem});
  }

  // Rebuild the path from the innermost updated slice back to the outer array.
  auto nestedArray = _array_read_impl(arr, indices.drop_back());
  auto updatedNested =
      _array_write_impl(nestedArray, indices.take_back(1), elem);
  return _array_write_impl(arr, indices.drop_back(), updatedNested);
}

cvc5::Term ImplicationTerm::buildTerm(cvc5::TermManager &mgr) {
  llvm::SmallVector<cvc5::Term> consequentTerms{consequents.begin(),
                                                consequents.end()};
  return mgr.mkTerm(cvc5::Kind::IMPLIES, {conjunctAll(antecedents, mgr),
                                          conjunctAll(consequentTerms, mgr)});
}

void ImplicationTerm::substitute(cvc5::Term oldTerm, cvc5::Term newTerm) {
  for (auto &antecedent : antecedents) {
    antecedent = antecedent.substitute(oldTerm, newTerm);
  }
  for (auto &consequent : consequents) {
    consequent = consequent.substitute(oldTerm, newTerm);
  }
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
  ensure(!conjuncts.empty(), "cannot build term from empty conjunction");

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
