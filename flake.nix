{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        ffmpeg-full = pkgs.ffmpeg_6-full;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            ffmpeg-full
            uv
            python313
          ];

          shellHook = ''
            export FFMPEG_PATH="${ffmpeg-full}/bin/ffmpeg"
            export FFPROBE_PATH="${ffmpeg-full}/bin/ffprobe"
            export PATH="${ffmpeg-full}/bin:$PATH"
            echo "ffmpeg: $(ffmpeg -version | head -1)"
            echo "libvmaf: $(${ffmpeg-full}/bin/ffmpeg -filters 2>&1 | grep -c libvmaf) filters available"
          '';
        };
      });
}
