# Why only `data/` is here

PhotoLab uses the `lensfunpy` **Python bindings** (installed via pip — see
`docs/requirements.txt`), not a compiled copy of the Lensfun C++ library.
The only thing PhotoLab reads from this folder at runtime is the lens
correction database in `data/db/` (used by `app_paths.lensfun_db_paths()`
and `imaging.try_lensfun_correct()`).

The upstream Lensfun C++ source, docs, build system, and test suite are
**not** vendored here — they aren't needed to run PhotoLab and just bloat
the repo (previously ~5MB of the ~10MB vendored tree). If you need to
update the lens database, replace `data/db/` with a fresh export from
https://github.com/lensfun/lensfun (`data/db`), or `pip install -U
lensfunpy` and point `app_paths` at its bundled database instead of
vendoring one at all.
