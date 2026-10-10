# Spectra from XASLIB

Transmission spectra from the IXAS X-ray Absorption Data Library,
[XASLIB](https://xaslib.xrayabsorption.org), downloaded on 2026-10-09. They are the XDI files
the library serves at `/rawfile/<id>/<name>.xdi`, unchanged, one directory per absorbing
element. Study 8 of `notebooks/moving_block_holdout_bootstrap.ipynb` uses them as measured
spectra with known answers.

| directory | spectra | used in Study 8 |
|---|---|---|
| `As/` | As2O3, As2O5, As2S3, AsS and GaAs; SSRL 2-3 and 4-1, 1997; 10 K, 100 K and room temperature | 100 K scans; 10 K scans as the other-session references |
| `Cr/` | Cr2O3, Cr2S3, CrO2, K2Cr2O7, K2CrO4, Na2CrO4; SSRL 2-3 and 4-1, 1997; 15 K, 100 K and room temperature | room-temperature SSRL 2-3 scans; 15 K scans as the other-session references |
| `Mn/` | MnO, Mn2O3, Mn3O4, MnO2; SSRL 4-3, 1995 | all |
| `Ni/` | Ni2O3, Ni(OH)2, NiS (SSRL 4-1, 1997) and Ni metal (APS 13-ID) | the oxide, hydroxide and sulfide |
| `Zn/` | ZnC2O4, ZnSO4, ZnS, hopeite, smithsonite, sphalerite and Zn foil (APS 13-BM-D, 2006), ZnSe (APS 13-BM-D); ZnO and Zn foil (APS 13-ID-E) | the six compounds measured in 2006 |

Most compounds have three scans from one session, which is what makes them useful: one scan is
a reference, and the other two are independent measurements of anything built from it.

- **Not normalized.** The files hold raw counts (`i0`, `itrans`, and `irefer` for the reference
  channel). The column order differs between files, so read them by the names in their
  headers. The notebook's
  `read_xdi`, `load_xaslib_series` and `normalize_xas` do this, calibrate each scan's energy on
  its reference channel, and normalize the edge step to 1.
- **License:** Creative Commons Zero (public domain). The library's
  [terms](https://docs.xrayabsorption.org/xaslib/license.html) ask that the collection be cited
  by its URL, https://xaslib.xrayabsorption.org, as the work of the International X-ray
  Absorption Society.
