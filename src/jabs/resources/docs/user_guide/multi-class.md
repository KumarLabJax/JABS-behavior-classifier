# Multi-Class Classification (Preview)

> **Preview feature.** Multi-class mode is under active development and is provided
> as a preview. Some capabilities are not yet available, and its behavior, stored
> data, and settings may change in upcoming JABS releases. Binary mode (the
> default) is unaffected.

## Overview

By default, JABS trains one **binary** classifier per behavior: each classifier
predicts whether a given frame contains that behavior or not, and behaviors are
independent of one another.

**Multi-class mode** instead trains a *single* classifier across all annotated
behaviors at once. Each frame is assigned to exactly one class: one of your
behaviors, or the reserved **None** (background) class. This is appropriate when
your behaviors are **mutually exclusive** - that is, an animal cannot be doing two
of them on the same frame.

## Enabling multi-class mode

Open **Project Settings** and set **Classifier Mode** to **Multi-class (Preview)**.
The setting is stored with the project, and the default for all projects remains
**Binary**.

Switching an existing project to multi-class mode is blocked if any frames are
labeled with two or more behaviors simultaneously; JABS lists the conflicting
videos so the overlaps can be resolved first.

## Labeling for multi-class

- Label each behavior as usual. Because classes are mutually exclusive, labeling a
  frame with one behavior clears any other behavior label on that frame.
- The **None** button records an explicit *background* label - frames that are
  none of your behaviors. In multi-class mode these explicit negatives are stored
  on a reserved **None** track rather than as "not behavior" on an individual
  behavior. The **Label Summary** reflects this: it shows the selected behavior's
  count and a **None** count (instead of "Behavior" / "Not Behavior").
- Only explicitly labeled frames (a behavior or **None**) are used for training;
  unlabeled frames are ignored.

## Known limitations (preview)

- **No prediction post-processing.** The post-processing step available for binary
  predictions is not yet applied to multi-class predictions. Multi-class
  predictions are shown and saved as raw (argmax) results only.
- **Project-level training settings.** Window size and label balancing apply at the
  project level for the single shared classifier rather than per behavior. Some
  per-behavior options available in binary mode (for example, selective symmetric
  augmentation per behavior) are not yet available in multi-class mode.
- **Mutual exclusivity required.** Behaviors must not overlap on the same frame.
  Overlapping labels must be resolved before switching to multi-class mode or
  training.
- **Migration.** Existing binary classifiers are not converted to multi-class
  format (or vice versa); the two modes maintain separate classifier and
  prediction files within a project.
- **Format stability.** The on-disk representation and available settings for
  multi-class mode may change in future releases.

## Command-line use

`jabs-classify` auto-detects whether a saved classifier is binary or multi-class
and dispatches accordingly, so existing command-line workflows continue to work
with multi-class classifiers without additional flags.