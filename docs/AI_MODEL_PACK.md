# PhotoLab AI Restoration Model Pack

PhotoLab does not bundle neural-network weights. Configure an external folder under
**Tools → Configuration / INI Editor → AI Restoration Model Pack**. The folder must
contain `photolab-model-pack.json`:

```json
{
  "format": "photolab-ai-model-pack-1",
  "name": "My Local Models",
  "providers": [
    {
      "id": "local_colorizer",
      "name": "Local Colorizer",
      "capabilities": ["colorize", "enhance"],
      "command": ["run_model.py", "--input", "{input}", "--output", "{output}",
                  "--operation", "{capability}", "--fidelity", "{fidelity}",
                  "--candidate", "{candidate}"],
      "license": "Record the code and weight licenses here"
    }
  ]
}
```

Supported capabilities are `colorize`, `face_restore`, `reconstruct`, `enhance`, and
`super_resolution`. Commands run locally with the model-pack folder as their working
directory. Python commands use PhotoLab's Python interpreter. A provider must write a
readable image to `{output}` and return exit code zero.

Source runs use PhotoLab's active Python interpreter. A packaged PhotoLab build
discovers Python from `PHOTOLAB_MODEL_PACK_PYTHON`, `.venv\Scripts\python.exe` in
the pack, `python\python.exe` in the pack, or the system PATH, in that order.

Placeholders: `{input}`, `{output}`, `{capability}`, `{fidelity}`, `{candidate}`, and
`{root}`. Model authors should preserve orientation and color encoding, tile large
images where necessary, publish code and weight licenses, and never silently upload
photographs. PhotoLab labels all provider results as AI-generated interpretations.

## Recommended Windows pack

The companion folder `C:\Users\brian\Documents\GitHub\photo_lab_ai_model_pack`
contains installers and adapters for the official Real-ESRGAN NCNN/Vulkan build and
the official DDColor paper-tiny model. It provides local colorization, enhancement,
and 4× super-resolution. Run `install_model_pack.ps1` and `install_ddcolor.ps1`,
then select that folder in PhotoLab's INI editor.

DDColor's code and selected checkpoint are both identified by their publishers as
Apache-2.0. Face reconstruction remains separate: CodeFormer's official S-Lab
license is non-commercial unless separately licensed.

## Restoration Studio workflow

1. Choose a local provider and operation, then click **Run Free Local AI Model
   (Offline)**. The progress window remains responsive while each candidate runs.
2. Completed candidates appear immediately in **Preview candidate** and in the main
   image preview. Use **View → Original** and **Restored Preview** to compare them.
3. Candidates are temporary by design. Use **Export Selected Candidate** to keep an
   unblended candidate, or **Apply & Save Copy** to save the selected candidate with
   the current restoration, blend, and mask settings.
4. Choose a destination in **Save Restored Copy**. A second progress window shows
   image encoding and byte-level writing progress. The original is never overwritten
   unless that exact filename is deliberately selected.
