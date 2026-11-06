# Hand reconstruction diagnosis

The previous finger-twist fix addressed motion geometry. This stage tested appearance using the same existing experiment branch and free Kaggle T4 session.

## Findings

The bundled fictional target reference is only 504 by 840 pixels and shows both hands tucked into its shorts pockets. Most finger surfaces are not visible. This reference is unsuitable for verifying faithful hand reconstruction.

A bounded GPU test rendered source frames 7, 24 and 48 through both the raw Gaussian-splatting renderer and the neural refinement renderer. The two focused modes used the **same model instance and reconstruction cache**, pose and camera. There were six rendered outputs and 4,671 recorded CUDA compute events; cleanup released the cache/model and left 6,144 allocated CUDA bytes.

The raw render shows some separate finger structure, but its surface appearance is speckled and unsuitable as the finished output. The focused neural rendering has some additional contour detail compared with enlarging the existing full view, but still merges fingers. Neither output meets the visual hand-quality requirement. These visual observations do not prove that reference occlusion is the only cause.

The enlarged full-view baseline was reconstructed in a separate run. Only the focused neural-versus-GS comparison has a strict same-cache guarantee. The focused crop excludes parts of the body and is a diagnostic, not a replacement full-body video.

## Changes

- Added `scripts/hand_reconstruction_diagnostic.py`: three fixed source poses, paired renderer outputs, input validation, one cached reconstruction, CUDA execution evidence, finite-output screening and cleanup verification.
- Added optional `output_renderer='neural'` to the backend's render method. The existing worker calls and default neural behavior are unchanged. Raw GS is available for diagnosis only; it is not selected automatically.
- Python compilation passed. Both rendering modes executed on the T4. No new model or paid resource was started.

## Next input and acceptance check

Use one sharp, authorized target reference showing the full body, both hands outside pockets, fingers visibly separated, and no object covering the hands. Keep the desired face, hair and clothing visible. A single front view still provides no ground truth for hidden surfaces or arbitrary views.

Reconstruct that reference and repeat the same three driving poses before a new 15-second test. Compare both silhouette and individual finger separation; reject improvements that only enlarge or sharpen fused fingers. If it still fails, evaluate a reconstruction model or hand-refinement stage with explicit hand-detail support rather than claiming that motion smoothing solves appearance.

Master and the original desktop application remain unchanged. Lip sync and live streaming remain unverified for this avatar pipeline.
