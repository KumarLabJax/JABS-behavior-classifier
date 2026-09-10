# Prediction Postprocessing

JABS has several postprocessing options to refine behavior predictions after classification. These options help reduce noise and can improve the quality of predicted behavior bouts.

To configure postprocessing, see the Tools→Prediction Postprocessing menu in the JABS GUI. The dialog has built-in help text for each option, see the dialog to explore available options.

Postprocessing settings are saved per behavior in the project settings, so different behaviors can use different postprocessing configurations. By default, no postprocessing is applied.

## Step Ordering

Postprocessing is implemented as a pipeline of steps, which are applied in order. The order of steps is currently fixed, but each step can be enabled or disabled independently. Future versions of JABS may allow reordering of steps.

## Evaluating Postprocessing with Cross-Validation

Postprocessing changes the predictions, so it also changes how well those predictions match your labels. To measure that, enable **Evaluate in Cross-Validation** in the Tools→Prediction Postprocessing dialog. Each time you train that behavior, the training report will then show a second set of cross-validation metrics with the enabled postprocessing steps applied, next to the raw metrics.

Because postprocessing steps reason about contiguous bouts, the held-out animal's *entire* track is predicted before the steps are applied — the same way predictions are generated when you classify. Metrics are then computed only on the labeled frames, where ground truth exists. This is what makes the comparison meaningful: steps that depend on the gaps between bouts, or on frames with no prediction at all, behave exactly as they would at prediction time.

A few things to keep in mind:

- Training is slower with this enabled, because every held-out animal's full track is predicted in addition to being trained on. The added cost is roughly one classification pass over the labeled animals. The first training run after a feature cache is cleared or invalidated is slower still, since the full-track features have to be computed rather than read from the cache.
- Like the postprocessing settings themselves, this option is saved per behavior.
- Postprocessing is only available for binary classifiers, so this option has no effect in multi-class mode.
- If no postprocessing steps are enabled, the evaluation is skipped (the pipeline would not change anything) and the training report says so.

## Visualizing Postprocessed Predictions

After applying postprocessing, JABS allows you to visualize the effects directly in the GUI. When viewing predictions in the Prediction Timeline, you can toggle between raw and postprocessed predictions to see how postprocessing affects the results.

## Saved Postprocessed Predictions

When postprocessing is applied, the postprocessed predictions are saved in the prediction H5 file under the dataset `predicted_class_postprocessed`. This allows you to retain both the raw and postprocessed predictions for future analysis.
