# Prediction Postprocessing

JABS has several postprocessing options to refine behavior predictions after classification. These options help reduce noise and can improve the quality of predicted behavior bouts.

To configure postprocessing, see the Tools→Prediction Postprocessing menu in the JABS GUI. The dialog has built-in help text for each option, see the dialog to explore available options.

Postprocessing settings are saved per behavior in the classifier file, so different behaviors can use different postprocessing configurations. By default, no postprocessing is applied.

## Step Ordering

Postprocessing is implemented as a pipeline of steps, which are applied in order. The order of steps is currently fixed, but each step can be enabled or disabled independently. Future versions of JABS may allow reordering of steps.

## Visualizing Postprocessed Predictions

After applying postprocessing, JABS allows you to visualize the effects directly in the GUI. When viewing predictions in the Prediction Timeline, you can toggle between raw and postprocessed predictions to see how postprocessing affects the results.

## Saved Postprocessed Predictions

When postprocessing is applied, the postprocessed predictions are saved in the prediction H5 file under the dataset `predicted_class_postprocessed`. This allows you to retain both the raw and postprocessed predictions for future analysis.
