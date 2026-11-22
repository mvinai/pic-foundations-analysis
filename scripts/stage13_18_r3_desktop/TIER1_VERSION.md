# Desktop Tier 1 package version

This folder ships:

- `r3a_desktop_full.py` and `test_r3a_desktop_full.py` — **v2.0.1-desktop**
- `r3b_desktop_full.py` and `test_r3b_desktop_full.py` — **v2.0.2-desktop** (v2.0.1 settings plus a hotfix that disables cuDNN only around the expected-gradients backward pass, because the cuDNN GRU backward does not run when the frozen network is in eval mode)

These public copies may differ slightly from the private desktop packages (see `docs/PUBLIC_REDACTIONS.md`). Tier 2 (`r4_...` v1.0.1) and Tier 3 (`r5_...` v1.0.1) are the released packages and are unchanged.
