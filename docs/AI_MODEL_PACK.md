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

Placeholders: `{input}`, `{output}`, `{capability}`, `{fidelity}`, `{candidate}`, and
`{root}`. Model authors should preserve orientation and color encoding, tile large
images where necessary, publish code and weight licenses, and never silently upload
photographs. PhotoLab labels all provider results as AI-generated interpretations.
