/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/VerificationUtils.h"

#include <fcntl.h>
#include <fstream>
#include <optional>
#include <errno.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Analysis/LightweightSignalEquivalenceAnalysis.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Transforms/LLZKComputeConstrainToProductPass.h>
#include <llzk/Transforms/LLZKTransformationPasses.h>
#include <llzk/Util/ErrorHelper.h>
#include <mlir/Pass/PassManager.h>

using namespace mlir;
using namespace llzk;

namespace lleq {
namespace {

std::optional<std::string> resolveSolverPath(llvm::StringRef envVar,
                                             llvm::StringRef defaultName) {
  if (std::optional<std::string> envPath = llvm::sys::Process::GetEnv(envVar)) {
    if (llvm::sys::fs::can_execute(*envPath)) {
      return *envPath;
    }
    llvm::report_fatal_error(StringRef{envVar} + " is set to '" + *envPath +
                             "', but that path is not executable");
  }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  auto solverPath = llvm::sys::findProgramByName(defaultName);
#pragma clang diagnostic pop
  if (!solverPath) {
    return std::nullopt;
  }
  return *solverPath;
}

std::string requireSolverPath(llvm::StringRef envVar, llvm::StringRef solver) {
  auto path = resolveSolverPath(envVar, solver);
  if (path) {
    return *path;
  }

  llvm::report_fatal_error("could not find an executable " + solver +
                           " binary; install " + solver + " or set " + envVar +
                           " to the full path of the solver binary");
}

struct SolverTempFiles {
  llvm::SmallString<128> stdinPath;
  llvm::SmallString<128> stdoutPath;

  SolverTempFiles() = default;
  SolverTempFiles(const SolverTempFiles &) = delete;
  SolverTempFiles &operator=(const SolverTempFiles &) = delete;

  SolverTempFiles(SolverTempFiles &&other) noexcept
      : stdinPath(std::move(other.stdinPath)),
        stdoutPath(std::move(other.stdoutPath)) {
    other.stdinPath.clear();
    other.stdoutPath.clear();
  }

  SolverTempFiles &operator=(SolverTempFiles &&other) noexcept {
    if (this == &other) {
      return *this;
    }

    llvm::sys::fs::remove(stdinPath);
    llvm::sys::fs::remove(stdoutPath);
    stdinPath = std::move(other.stdinPath);
    stdoutPath = std::move(other.stdoutPath);
    other.stdinPath.clear();
    other.stdoutPath.clear();
    return *this;
  }

  ~SolverTempFiles() {
    llvm::sys::fs::remove(stdinPath);
    llvm::sys::fs::remove(stdoutPath);
  }
};

SolverTempFiles createSolverTempFiles(llvm::StringRef query) {
  int stdinFd = -1;
  SolverTempFiles files;
  llzk::ensure(llvm::sys::fs::createTemporaryFile("lleq-solver-stdin", "smt2",
                                                  stdinFd, files.stdinPath) ==
                   std::error_code{},
               "failed to create temporary file for solver stdin");

  int stdoutFd = -1;
  llzk::ensure(llvm::sys::fs::createTemporaryFile("lleq-solver-stdout", "txt",
                                                  stdoutFd, files.stdoutPath) ==
                   std::error_code{},
               "failed to create temporary file for solver stdout");

  std::ofstream os{files.stdinPath.c_str()};
  os.write(query.data(), query.size());
  os.close();

  if (stdinFd >= 0) {
    ::close(stdinFd);
  }
  if (stdoutFd >= 0) {
    ::close(stdoutFd);
  }

  return files;
}

std::string readSolverOutput(llvm::StringRef path) {
  std::ifstream is{path.str().c_str()};
  std::string result{std::istreambuf_iterator<char>{},
                     std::istreambuf_iterator<char>{}};
  if (is) {
    is.seekg(0);
    result.assign(std::istreambuf_iterator<char>{is},
                  std::istreambuf_iterator<char>{});
  }
  return result;
}

SolverResultKind classifySolverOutput(llvm::StringRef output) {
  output = output.ltrim();
  if (output.consume_front("unsat")) {
    return SolverResultKind::Unsat;
  }
  if (output.consume_front("sat")) {
    return SolverResultKind::Sat;
  }
  if (output.consume_front("unknown")) {
    return SolverResultKind::Unknown;
  }
  return SolverResultKind::ParseFailure;
}

struct RunningSolverJob {
  SolverInvocationSpec spec;
  SolverTempFiles files;
  llvm::SmallVector<std::string> ownedArgs;
  pid_t pid = -1;
  bool finished = false;
};

RunningSolverJob spawnSolverJob(const SolverInvocationSpec &spec,
                                llvm::StringRef query) {
  RunningSolverJob job;
  job.spec = spec;
  job.files = createSolverTempFiles(query);

  job.ownedArgs = spec.args;
  job.ownedArgs.reserve(spec.args.size() + (spec.passQueryFileAsArg ? 1 : 0));
  if (spec.passQueryFileAsArg) {
    job.ownedArgs.push_back(job.files.stdinPath.str().str());
  }
  llvm::SmallVector<llvm::StringRef> argv;
  argv.reserve(job.ownedArgs.size());
  for (auto &arg : job.ownedArgs) {
    argv.push_back(arg);
  }

  job.pid = ::fork();
  if (job.pid < 0) {
    llvm::report_fatal_error("failed to fork solver process");
  }

  if (job.pid == 0) {
    if (::setpgid(0, 0) < 0) {
      _exit(127);
    }

    int stdinFd = spec.passQueryFileAsArg
                      ? ::open("/dev/null", O_RDONLY)
                      : ::open(job.files.stdinPath.c_str(), O_RDONLY);
    int stdoutFd = ::open(job.files.stdoutPath.c_str(), O_WRONLY | O_TRUNC);
    if (stdinFd < 0 || stdoutFd < 0) {
      _exit(127);
    }
    if (::dup2(stdinFd, STDIN_FILENO) < 0 ||
        ::dup2(stdoutFd, STDOUT_FILENO) < 0) {
      _exit(127);
    }

    ::close(stdinFd);
    ::close(stdoutFd);

    std::vector<char *> execArgv;
    execArgv.reserve(job.ownedArgs.size() + 1);
    for (auto &arg : job.ownedArgs) {
      execArgv.push_back(arg.data());
    }
    execArgv.push_back(nullptr);
    ::execv(spec.path.c_str(), execArgv.data());
    _exit(127);
  }

  // Put the solver in its own process group so cancellation can kill any
  // descendants it might spawn before the parent observes completion.
  if (::setpgid(job.pid, job.pid) < 0 && errno != EACCES && errno != ESRCH) {
    llvm::report_fatal_error("failed to assign solver process group");
  }

  return job;
}

SolverRunResult collectFinishedJob(RunningSolverJob &job) {
  std::string output = readSolverOutput(job.files.stdoutPath);
  return SolverRunResult{classifySolverOutput(output), std::move(output),
                         job.spec.name};
}

void killJob(RunningSolverJob &job) {
  if (!job.finished && job.pid > 0) {
    if (::kill(-job.pid, SIGKILL) < 0 && errno != ESRCH) {
      llvm::report_fatal_error("failed to kill solver process group");
    }
  }
}

void reapJob(RunningSolverJob &job) {
  if (job.finished) {
    return;
  }

  int status = 0;
  while (::waitpid(job.pid, &status, 0) < 0) {
    if (errno == EINTR) {
      continue;
    }
    if (errno == ECHILD) {
      break;
    }
    llvm::report_fatal_error("failed to reap solver process");
  }
  job.finished = true;
}

} // namespace

llvm::LogicalResult ensureProductFunc(ModuleOp module,
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

  if (llvm::failed(aligner.alignCalls(productFunc))) {
    return llvm::failure();
  }

  // Now, try fusing loops
  PassManager pm{module->getContext()};
  pm.addPass(llzk::createFuseProductLoopsPass());
  return pm.run(module);
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

llvm::FailureOr<std::string>
invokeSolverOnQuery(llvm::StringRef solverPath,
                    llvm::ArrayRef<llvm::StringRef> args, llvm::StringRef query,
                    bool passQueryFileAsArg) {
  auto files = createSolverTempFiles(query);

  llvm::SmallVector<llvm::StringRef> commandArgs(args.begin(), args.end());
  if (passQueryFileAsArg) {
    commandArgs.push_back(files.stdinPath.str());
  }

  std::string error;
  auto code = llvm::sys::ExecuteAndWait(
      solverPath, commandArgs,
      /*Env=*/std::nullopt,
      /*Redirects=*/
      {passQueryFileAsArg ? "" : std::string{files.stdinPath},
       std::string{files.stdoutPath}, ""},
      0, 0, &error);
  if (code) {
    llvm::report_fatal_error(error.c_str());
  }

  return readSolverOutput(files.stdoutPath);
}

llvm::FailureOr<SolverRunResult>
invokeSolverPortfolio(llvm::ArrayRef<SolverInvocationSpec> solvers,
                      llvm::StringRef query) {
  llzk::ensure(!solvers.empty(), "solver portfolio must not be empty");

  llvm::SmallVector<RunningSolverJob, 0> jobs;
  jobs.reserve(solvers.size());
  for (auto &solver : solvers) {
    jobs.push_back(spawnSolverJob(solver, query));
  }

  std::optional<SolverRunResult> lastNonConclusive;
  auto cleanup = [&jobs]() {
    for (auto &job : jobs) {
      killJob(job);
    }
    for (auto &job : jobs) {
      reapJob(job);
    }
  };

  while (true) {
    bool anyRunning = false;
    for (auto &job : jobs) {
      if (job.finished) {
        continue;
      }

      int status = 0;
      auto waited = ::waitpid(job.pid, &status, WNOHANG);
      if (waited == 0) {
        anyRunning = true;
        continue;
      }
      if (waited < 0) {
        job.finished = true;
        lastNonConclusive =
            SolverRunResult{SolverResultKind::ExecutionFailure,
                            "failed to wait for solver process", job.spec.name};
        continue;
      }

      job.finished = true;
      auto result = collectFinishedJob(job);
      if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        result.kind = SolverResultKind::ExecutionFailure;
      }
      if (result.kind == SolverResultKind::Sat ||
          result.kind == SolverResultKind::Unsat) {
        cleanup();
        return result;
      }
      lastNonConclusive = std::move(result);
    }

    if (!anyRunning) {
      cleanup();
      if (lastNonConclusive) {
        return *lastNonConclusive;
      }
      return llvm::failure();
    }

    ::usleep(1000);
  }
}

llvm::SmallVector<SolverInvocationSpec> getWeakestPreconditionPortfolio() {
  llvm::SmallVector<SolverInvocationSpec> solvers;
  if (auto z3 = resolveSolverPath("LLEQ_Z3", "z3")) {
    solvers.push_back(SolverInvocationSpec{
        .name = "z3",
        .path = *z3,
        .args = llvm::SmallVector<std::string>{*z3, "-smt2"},
        .passQueryFileAsArg = true,
    });
  }
  if (auto cvc5 = resolveSolverPath("LLEQ_CVC5", "cvc5")) {
    solvers.push_back(SolverInvocationSpec{
        .name = "cvc5",
        .path = *cvc5,
        .args = llvm::SmallVector<std::string>{*cvc5},
        .passQueryFileAsArg = false,
    });
  }

  llzk::ensure(!solvers.empty(),
               "could not find an executable z3 or cvc5 binary; install one "
               "or set LLEQ_Z3/LLEQ_CVC5 to full solver paths");
  return solvers;
}

bool checkUnsatWithZ3(llvm::StringRef query) {
  auto solverPath = requireSolverPath("LLEQ_Z3", "z3");
  llvm::SmallVector<llvm::StringRef> args{solverPath, "-smt2"};
  auto output = invokeSolverOnQuery(solverPath, args, query,
                                    /*passQueryFileAsArg=*/true);
  if (failed(output)) {
    return false;
  }

  llvm::StringRef result{*output};
  result = result.ltrim();
  return result.consume_front("unsat");
}

bool checkUnsatWithPortfolio(llvm::StringRef query) {
  auto result = invokeSolverPortfolio(getWeakestPreconditionPortfolio(), query);
  if (failed(result)) {
    return false;
  }
  return result->kind == SolverResultKind::Unsat;
}

} // namespace lleq
