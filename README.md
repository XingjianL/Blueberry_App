
## Paper: Developing Automated Tools for Blueberry Count, Weight, and Size Estimation on a Mobile Device.

### This repo host the source code and tools used to derive the results and application.
<br>

The image data can be found here: [First and Second Dataset](https://drive.google.com/drive/folders/1NXhgfqKMEBnDkbVKodzXc4zurVZLxYua?usp=sharing), and [Third Dataset](https://drive.google.com/drive/folders/1AsFSRJXpXgysDeFijOYzRPWRZhQRah24?usp=sharing).

The measured data can be found here: [Sheet](https://docs.google.com/spreadsheets/d/13l3W_Z-OHitcX8nzLA1neZsaRfOizW1q/edit?usp=sharing&ouid=104460468007818024731&rtpof=true&sd=true)

The results are derived from the `/analysis` folder. Remember to change the folder paths in the scripts.

The trained yolov5-tflite models are in the `/app/ml` folder.

<br>

The application source code is in the `/app` folder, with the Android APK in `/app/bin`. 

Note for the app: This app has not been tested on actual samples nor the output results are used in the paper, except only used to measure time performance on a mobile device. Furthermore for area estimation data, only the pixel area is saved on device and it does not save the image used on device. I recommend using the original phone cameara, the scripts, and datasheets in `/analysis` to match the counting and weight estimation shown in the paper for minimal adjustments.

<br>

To install an APK, use adb tool, see more here https://stackoverflow.com/questions/7076240/install-an-apk-file-from-command-prompt. 

You could also build your own by following the Kivy Buildozer Tutorial https://kivy.org/doc/stable/guide/packaging-android.html. Install buildozer tool, and the `buildozer.spec` file for building is in `/app`. It will then generate the apk in `/bin`

