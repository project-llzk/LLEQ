{
  pkgs,
  stdenv, lib,

  # build dependencies
  clang, cmake, ninja,
  mlir, # nlohmann_json, do I need nlohmann??
  llzk,

  # test dependencies
  gtest, python3, lit, z3, # cvc5 ??
}:
stdenv.mkDerivation {
  name = "verifier";
  version = "0.1.0";
  src = 
    let
      src0 = lib.cleanSource (builtins.path {
        path = ./..;
        name = "verifier-source";
      });
    in
      lib.cleanSourceWith {
        filter = path: type: !(lib.lists.any (x: x) [
            (path == toString (src0.origSrc + "/README.md"))
            (type == "directory" && path == toString (src0.origSrc + "/.github"))
            (type == "regular" && lib.strings.hasSuffix ".nix" (toString path))
            (type == "regular" && baseNameOf path == "flake.lock")
          ]);
        src = src0;
      };
  
  nativeBuildInputs = [clang cmake ninja z3.lib z3];
  buildInputs = [mlir llzk z3.lib];

  preBuild = ''
    export LD_LIBRARY_PATH=${z3.lib}/lib:$LD_LIBRARY_PATH
  '';

  preConfigure = '' if [[ "$(uname)" == "Darwin" ]]; then export OLD_PATH=$PATH export PATH="$PATH:/usr/bin/" fi '';
  postConfigure = '' if [[ "$(uname)" == "Darwin" ]]; then export PATH=$OLD_PATH fi '';
}
