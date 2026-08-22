# PhotoLab scripts/

Drop Python scripts here. From **Tools → Run Script…** (or **Image → Run Script…**),
PhotoLab executes a selected file with:

```text
python your_script.py --path /full/image.jpg --recipe /tmp/recipe.json
```

Environment variables (also set):

- `PHOTOLAB_IMAGE` — current image path  
- `PHOTOLAB_RECIPE_JSON` — path to a JSON dump of the current Recipe  

Example: `export_caption.py` writes a sidecar `.txt` with the filename and exposure.
