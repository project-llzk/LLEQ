# Based on the LLZK flake
{
  inputs = {
    llzk-pkgs.url = "github:Veridise/llzk-nix-pkgs?ref=main";

    nixpkgs = {
      # url = "github:NixOS/nixpkgs";
      follows = "llzk-pkgs/nixpkgs";
    };

    flake-utils = {
      # url = "github:numtide/flake-utils/v1.0.0";
      follows = "llzk-pkgs/flake-utils";
    };

    llzk = {
      url = "github:project-llzk/llzk-lib?ref=main";
      inputs = {
        nixpkgs.follows = "llzk-pkgs/nixpkgs";
        flake-utils.follows = "llzk-pkgs/flake-utils";
        llzk-pkgs.follows = "llzk-pkgs";
      };
    };

    release-helpers.follows = "llzk/release-helpers";
  };

  # Custom colored bash prompt
  nixConfig.bash-prompt = ''\[\e[0;32m\][LLEQ]\[\e[m\] \[\e[38;5;244m\]\w\[\e[m\] % '';

  outputs = { self, nixpkgs, flake-utils, llzk-pkgs, release-helpers, llzk }:
    {
      # First, we define the packages used in this repository/flake
      overlays.default = final: prev: {

        # Default lleq build uses the default compiler for the system (usually gcc for Linux and clang for Macos)
        lleq = final.callPackage ./nix/lleq.nix { clang = final.clang_20; llzk = final.llzk; };
        lleq-debug = final.callPackage ./nix/lleq.nix { 
          clang = final.clang_20;
          llzk = final.llzk-debug;
          # mlir_pkg = final.mlir-debug;
          # cmakeBuildType = "Debug";
        };
        # Build in release with symbols mode with a particular compiler and sanitizers enabled.
        # Mostly useful for development and CI
        lleqClang = (final.lleq.override { stdenv = final.clangStdenv; }).overrideAttrs(attrs: {
          cmakeBuildType = "RelWithDebInfo";
        });
        lleqGCC = (final.lleq.override { stdenv = final.gccStdenv; }).overrideAttrs(attrs: {
          cmakeBuildType = "RelWithDebInfo";
        });
      };
    } //
    (flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;

          overlays = [
            self.overlays.default
            llzk-pkgs.overlays.default
            release-helpers.overlays.default
            llzk.overlays.default
          ];
        };
      in
      {
        # Now, we can define the actual outputs of the flake
        packages = flake-utils.lib.flattenTree {
          # Copy the packages from the overlay.
          inherit (pkgs) lleq-debug lleq;

          # For debug purposes, expose the MLIR/LLVM packages.
          inherit (pkgs) mlir llzk clang gtest python3 lit z3 cvc5;
          # Prevent use of libllvm and llvm from nixpkgs, which will have different
          # versions than the mlir from llzk-pkgs.
          inherit (pkgs.llzk-llvmPackages) libllvm llvm;

          default = pkgs.lleq-debug;
          withClang = pkgs.lleqClang;
          withGCC = pkgs.lleqGCC;
        };

        devShells = flake-utils.lib.flattenTree {
          default =  pkgs.lleq-debug.overrideAttrs (old: {
            nativeBuildInputs = (with pkgs; [
              doxygen
              git

              # clang-tidy and clang-format
              llzk-llvmPackages.clang-tools

              # git-clang-format
              libclang.python

            ]) ++ old.nativeBuildInputs;

            shellHook = ''
              # needed to get accurate compile_commands.json
              export CXXFLAGS="$NIX_CFLAGS_COMPILE"

              # Add binary dir to PATH for convenience
              export PATH="$PWD"/build/tools/lleq/:"$PATH"

              # Add release helpers to the PATH for convenience
              export PATH="${pkgs.changelogCreator.out}/bin":"$PATH"

              # Add samply to the PATH for profiling during development
              export PATH="${pkgs.samply.out}/bin":"$PATH"

              # Add LLDB to the PATH for debugging
              #export PATH="${pkgs.lldb.out}/bin":"$PATH"

              # For using mlir-tblgen inside the dev environment
              export LD_LIBRARY_PATH=${pkgs.z3.lib}/lib:$LD_LIBRARY_PATH

              # Disable container overflow checks because it can give false positives in
              # ConvertZmlToLlzkPass::runOnOperation() since LLVM itself is not built with ASan.
              # https://github.com/google/sanitizers/wiki/AddressSanitizerContainerOverflow#false-positives
              export ASAN_OPTIONS=detect_container_overflow=0:detect_leaks=0
            '';
          });

          llvm = pkgs.mkShell {
            buildInputs = [ pkgs.libllvm.dev ];
          };
        };
      }
    ));
}
