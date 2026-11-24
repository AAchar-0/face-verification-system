# Face Verification Technical Reference

**Scope:** Design, privacy controls, evaluation method, testing and troubleshooting for the Consent-Based Face Verification System.
**Audience:** Developers and reviewers maintaining or assessing the project.
**Last updated:** November 2026

## Contents

1. [Design goals](#1-design-goals)
2. [How verification works](#2-how-verification-works)
3. [Privacy and security controls](#3-privacy-and-security-controls)
4. [Template lifecycle](#4-template-lifecycle)
5. [Evaluation method and results](#5-evaluation-method-and-results)
6. [Testing](#6-testing)
7. [Common pitfalls and resolutions](#7-common-pitfalls-and-resolutions)
8. [Limitations](#8-limitations)
9. [Release checklist](#9-release-checklist)

---

## 1. Design goals

The system performs **one-to-one verification** ("is this the person who enrolled?"), not one-to-many identification. Goals, in priority order: consent before capture, minimal data retained, protection of what is retained, and accuracy that is measured and reported honestly.

## 2. How verification works

| Stage | Method |
|---|---|
| Detection | YuNet finds faces and five landmarks (eyes, nose, mouth corners) |
| Quality gate | Exactly one face, at least 80 pixels across, detector confidence at least 0.90 |
| Alignment | The face is straightened and cropped using the landmarks |
| Embedding | SFace turns the aligned face into 128 numbers |
| Template | Several embeddings are length-normalised, averaged and normalised again |
| Comparison | Cosine similarity between the template and each camera frame |
| Decision | The **median** similarity over at least three clear frames must reach the threshold |

The median prevents a single odd frame from deciding the outcome, and requiring several frames prevents one lucky frame from being enough.

## 3. Privacy and security controls

| Control | Implementation | Verified by |
|---|---|---|
| Consent before capture | The exact phrase `I AGREE` is required before the camera is opened | Automated tests: the camera is not opened without consent, and opens only after the consent answer |
| No images stored | Frames are used in memory and never written | Test: enrollment produces exactly one `.enc` file and nothing else |
| Encryption at rest | Fernet: AES (Advanced Encryption Standard) encryption with an HMAC (Hash-based Message Authentication Code) seal | Tests: file contents are unreadable; a wrong key or one flipped bit is refused |
| Key handling | The key is stored in `.env`, which Git ignores | `.gitignore` rules; `git status` review before every commit |
| Retention | Templates expire (90 days by default); expired ones are deleted on use or by `purge` | Tests for expiry and purge |
| Deletion | `delete` overwrites then removes the file | Tests |
| Session-only mode | Templates live in RAM (Random Access Memory) and are zeroed at exit, even after an interruption | Tests: nothing is written to disk; memory is erased on interruption |
| Safe file names | Names are limited to letters, digits, hyphens and underscores | Tests reject `../evil` and similar |

## 4. Template lifecycle

`enroll` (consent, capture, average, encrypt, save) leads to `verify` (load, compare, decide), then expiry or `delete`. In `demo` mode the same steps run entirely in memory and everything is erased when the session ends. The live window is the control panel during a demo: V, Enter or Space verifies, and Q or Esc quits.

## 5. Evaluation method and results

- **Benchmark:** Labeled Faces in the Wild (LFW), standard protocol: 10 folds, each with 300 same-person and 300 different-person pairs. Images stay local and are never committed.
- **Metrics:** FAR (False Accept Rate: different people wrongly accepted), FRR (False Reject Rate: the same person wrongly rejected), EER (Equal Error Rate: where the two are equal), and accuracy.
- **Held-out measurement:** for each fold, the threshold is chosen on the other nine and measured on the tenth.
- **Intervals:** 95% Wilson confidence ranges, which remain meaningful when a count is zero or very small.
- **Deployed threshold:** 0.359, chosen so that at most 0.1% of different-person pairs are accepted.

| Result | Value |
|---|---|
| Accuracy (standard protocol) | 99.10% (± 0.48%) |
| EER | 1.16% |
| FAR at the deployed threshold | 0.068% (2 of 2,944; 95% range 0.019% to 0.247%) |
| FRR at the deployed threshold | 1.90% (56 of 2,941) |
| End-to-end FRR including pairs with no detectable face | 3.83% (115 of 3,000) |

With 2,944 different-person pairs the smallest measurable FAR is about 0.034%, so no claim such as "one in a million" can be supported.

## 6. Testing

```powershell
pytest
```

The suite covers matching maths, the encrypted store, the evaluation statistics, and the full enrollment, verification and demo flows using a fake camera and a fake face engine. The real camera and the real models are checked by hand with `python src/01_check_setup.py` and the `demo` command.

## 7. Common pitfalls and resolutions

| Symptom | Cause | Resolution |
|---|---|---|
| The camera window shows "(Not Responding)" and ignores keys | The program was waiting for terminal input while nothing refreshed the window | Keep refreshing the window during every wait, and make the window the control panel |
| Camera light on before consent | The camera was opened at program start | Open the camera only after consent; tests assert the order |
| Progress label stays at 4/5 | The label was drawn before the fifth frame was saved | Draw the label after saving the frame |
| Two environments appear (`.venv` and `.venv-1`) | The editor created a second environment | Select the project's interpreter, delete the extra folder, and ignore `.venv*/` |
| A model file is only a few hundred bytes | A Git LFS (Large File Storage) pointer was downloaded instead of the model | Size check in the download script |
| Benchmark site not found (DNS error) | The original host no longer resolves | Use the verified mirror and check each file's SHA-256 checksum |
| Held-out accuracy varies between folds | A threshold placed on the edge of the training data | Use the midpoint of the best range |
| Confidence range prints `1e-19` instead of 0 | Floating-point rounding | Return exact bounds at the extremes |
| Chart fails when all scores are identical | Zero-width histogram bins | Shared bins and a guard for zero range |
| Some benchmark pairs cannot be scored | The detector found no face in 70 of 7,701 photos | Report skipped pairs and the end-to-end FRR |
| OpenCV prints warning lines at start | A notice from its new inference engine | Harmless; the log level is lowered |

## 8. Limitations

- No liveness detection: a printed photo or a video could fool the system.
- Templates are not anonymous: embeddings can sometimes be partly reversed, so they are encrypted.
- The key is stored beside the data, which suits a demonstration. A real deployment needs a key vault.
- The threshold is calibrated on single benchmark photos, while the program compares an averaged template with several frames.
- LFW is skewed in who it includes, so results do not describe all groups equally. Fairness testing on a dataset with demographic labels is future work.
- The training data of the released SFace model is not documented by OpenCV.

## 9. Release checklist

1. `pytest` passes.
2. `git status` shows no `.env`, `data/`, `models/`, templates or images.
3. The README results match `reports/lfw_evaluation.json`.
4. The README and demo state that no photo is saved, and that this is not a production security system.
5. Only the author's own face appears in screenshots and recordings.
6. Volunteer data, if collected, has been deleted and signed forms are kept outside the repository.
