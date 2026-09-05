# PhotoLab Local AI Model Pack

This separate folder adds local, offline AI processing to PhotoLab without making
the normal PhotoLab download enormous or silently redistributing restricted models.

## Install the included providers

1. Right-click `install_model_pack.ps1` and choose **Run with PowerShell** for
   enhancement and super-resolution.
2. Right-click `install_ddcolor.ps1` and choose **Run with PowerShell** for
   automatic colorization.
3. Open PhotoLab's **Configuration / INI Editor**.
4. Set **AI Restoration Model Pack** to this folder.
5. Restart PhotoLab and open **Restoration Studio → AI Restoration Lab**.

The installer downloads the official 45 MB Real-ESRGAN Windows/Vulkan package.
Processing stays on this computer after installation. A Vulkan-capable graphics
driver is required; many integrated Intel, AMD, and NVIDIA laptop GPUs support it.
DDColor additionally requires Python with PyTorch, torchvision, NumPy, and OpenCV;
`diagnose_model_pack.py` reports whether they are available. Packaged PhotoLab builds
can use `PHOTOLAB_MODEL_PACK_PYTHON` to select a particular Python installation.

## Included operations

- **Enhance:** runs Real-ESRGAN and returns an image at the original dimensions.
- **Super Resolution:** returns a 4× image.
- **Colorize:** runs the official DDColor paper-tiny checkpoint locally. Candidate
  choices range from restrained to richer chroma while preserving source luminance.
- **Fidelity:** higher values blend back more of the original and preserve more grain.
- **Candidate results:** later candidates progressively increase the AI blend strength.

Always retain the original scan. AI enhancement can invent plausible texture and
should not be treated as documentary evidence.

## Diagnostics

Run `python diagnose_model_pack.py` in this folder. It reports whether the manifest
and engine are present and whether the executable can start.

## Face restoration and reconstruction

DDColor code and the selected official `ddcolor_paper_tiny` checkpoint are both
identified as Apache-2.0 by their publishers. CodeFormer is not bundled because
its official license restricts it to
non-commercial use. This separation keeps PhotoLab honest and redistributable.

## Files and privacy

The provider receives a temporary image from PhotoLab, invokes a local executable,
and writes a local result. It does not upload images or require an account.
