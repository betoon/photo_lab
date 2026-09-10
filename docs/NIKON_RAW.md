# Direct Nikon RAW support on Windows

PhotoLab tries rawpy first, then Nikon's Image SDK for NEF/NRW files it cannot decode.
The Nikon path develops the original file into an in-memory RGB image at 8-bit preview
or 16-bit export precision. It does not create DNG files or modify the NEF.
Nikon's as-shot rendering/white balance is baked into the developed RGB image.

## Setup

Obtain Nikon's NEF/NRW Image SDK separately and extract it. Select its `Image SDK`
folder in Tools > Configuration > Nikon Image SDK, or set `PHOTOLAB_NIKON_SDK`.
A sibling `nikon_sdk/Image SDK` checkout is discovered automatically.

Build the small native helper using Visual Studio C++ tools:

```powershell
./tools/build_nikon_decoder.ps1 -SdkRoot "C:/path/to/Image SDK"
```

Then run `python main.py`. Portable builds include the helper when built; the
Nikon SDK remains an external installation. `build_portable.bat` builds the helper
when the sibling SDK is present. A previously built PhotoLab.exe must be rebuilt
before it can use this feature.

The helper runs in an isolated process with Nikon's runtime files beside it in
LocalAppData/PhotoLab/nikon_runtime. The image buffer is passed through a temporary
file removed after loading. This arrangement is required by Nikon's RAW engine;
loading NkImgSDK.dll by absolute path alone can return a parameter error.
SDK binaries, color profiles, generated helper executables and photographs are
not committed to this repository. Follow Nikon's SDK license when deploying it.

Verified with SDK 1.46.0 and a Nikon Z6 III high-efficiency NEF at 6048 x 4032.
