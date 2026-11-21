# Consent-Based Face Verification System

> A Python and OpenCV project that checks whether a person is who they claim to be (one-to-one verification), built around explicit consent, no stored photos, encrypted templates, and honestly reported accuracy.

**Status:** In progress. Setup and the environment check are done. Enrollment, verification and evaluation are next.

## What it is, and what it is not

| It is | It is not |
|---|---|
| One-to-one verification: "is this the person who enrolled?" | One-to-many identification: searching a crowd or a database for who someone is |
| Opt-in: nothing is captured until the person has read and accepted a consent statement | Covert, passive or automatic capture |
| A technical demonstration with measured accuracy | A production security system or legal advice |

## Privacy and ethics by design

These principles are applied as each feature is built:

| Principle | Implementation |
|---|---|
| Explicit consent first | A consent statement must be accepted before any capture; the version and time are recorded |
| Data minimisation | No photos are saved. Only a 128-number template (an embedding) is kept |
| Encryption at rest | Templates are encrypted; the key lives outside the repository |
| Retention and deletion | Templates expire, and one command deletes a person's data |
| Safe test data | Only the author's own face and volunteers who consent. No faces are scraped from the web or social media |
| No public live camera demo | A hosted camera app would collect strangers' biometric data, so the demo runs locally |

## Models and licences

| Model | Task | Licence |
|---|---|---|
| YuNet | Face detection | MIT |
| SFace | Face recognition (128-number embeddings) | Apache 2.0 |

Both come from the OpenCV Zoo. **Provenance note:** OpenCV does not document what the released SFace model was trained on. The SFace paper used large research face datasets (CASIA-WebFace, VGGFace2 and MS1M). This project cannot verify the consent status of those images, and the licence above covers the model files only.

## Evaluation plan

- **Benchmark:** the standard Labeled Faces in the Wild (LFW) pair protocol, run locally with images that are never committed or redistributed.
- **Metrics:** false accept rate (FAR), false reject rate (FRR), equal error rate (EER) and the ROC curve, with the decision threshold chosen on training folds and tested on a held-out fold.
- **Caveat:** LFW consists of web-collected photos of public figures and is skewed in who it includes. High scores overstate real-world accuracy, especially across demographic groups. LFW has also been described as an ethically problematic dataset in published research, which is why it is used for evaluation only.

Huang, G. B., Ramesh, M., Berg, T., & Learned-Miller, E. (2007). *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments.* University of Massachusetts, Amherst, Technical Report 07-49.

## Legal note

This is a technical demonstration, not legal advice. Anyone deploying face verification needs their own legal review and a data protection impact assessment. In general terms:

- **Kenya:** biometric data is classed as sensitive personal data under the Data Protection Act 2019, and the Data Protection Commissioner has issued guidance on biometric data.
- **European Union:** the GDPR restricts processing biometric data to uniquely identify a person unless an exception applies, such as explicit consent.
- **Illinois (USA):** the Biometric Information Privacy Act requires notice, a written release before collection, and a public retention and destruction policy.

## Roadmap

- [x] Project setup, licence and environment check
- [ ] Enrollment with explicit consent and encrypted templates
- [ ] Verification and threshold selection
- [ ] Evaluation on LFW pairs with a FAR and FRR report
- [ ] Retention, expiry and one-command deletion
- [ ] Automated tests
- [ ] Technical reference and final write-up

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install opencv-python numpy cryptography matplotlib pytest
python src/download_models.py
python src/01_check_setup.py
```

## License and attribution

Released under the [MIT License](LICENSE). Reuse is welcome, and the license requires that the copyright notice stays with any copy or substantial portion of this work. Please credit it as:

> Anade A'Acharo, *Consent-Based Face Verification System*, https://github.com/AAchar-0/face-verification-system

Presenting this work as your own, or removing the notice, is not permitted by the license.
