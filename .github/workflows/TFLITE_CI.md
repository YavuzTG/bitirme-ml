# TFLite conversion CI

What the workflow does
- Installs Python 3.11 and dependencies from `masaustu/requirements.txt`.
- Runs `python masaustu/convert_to_tflite.py` to generate `model_cnn.tflite` (and `model_lstm.tflite` if available).
- Uploads produced TFLite files as workflow artifacts and attempts to commit them back to the repo.

How to trigger
- Push to `main`/`master`, or open this repository's Actions tab and run the "Convert Keras to TFLite" workflow (workflow_dispatch).

Where to fetch results
- If the workflow successfully committed the files they'll appear under `mobile_app/assets` in the repository.
- If commit failed, open the workflow run and download the artifact named `tflite-models` from the Actions UI.

Local helper
- If you downloaded artifact files to your machine, you can place them in `masaustu/` and run:
```
./tools/install_tflite_assets.sh
```
This copies `model_cnn.tflite` and `model_lstm.tflite` into `mobile_app/assets`, stages and commits them.

Next steps after artifacts are in repo
- Run `flutter pub get` in `mobile_app`, build the app and test on device. The app prefers on-device TFLite inference and falls back to the server if assets are missing.
