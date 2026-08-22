# Developer


### Testing (#18)

```bash
pip install pytest opencv-python-headless
PYTHONPATH=. pytest tests/ -q
```

- `tests/test_recipe.py` — Recipe to_dict / from_dict / JSON
- `tests/test_catalog.py` — upsert, migrate, roots
- `tests/test_apply_recipe_golden.py` — synthetic images, PSNR/SSIM floors
