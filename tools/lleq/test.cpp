#include <cvc5/cvc5.h>
#include <iostream>

cvc5::Term get_sum(cvc5::TermManager &mgr) {

  // Create two int variables `a`, `b`, and make a term representing `a + b`
  auto a = mgr.mkConst(mgr.getIntegerSort(), "a");
  auto b = mgr.mkConst(mgr.getIntegerSort(), "b");
  auto a_plus_b = mgr.mkTerm(cvc5::Kind::ADD, {a, b});

  return a_plus_b;
}

int main() {
  cvc5::TermManager mgr;
  auto term = get_sum(mgr);
  term = mgr.mkTerm(cvc5::Kind::MULT,
                    {term, mgr.mkConst(mgr.getIntegerSort(), "c")});
  std::cout << term << "\n"; // prints (+ a b)
}
