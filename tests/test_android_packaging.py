from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AndroidPackagingTest(unittest.TestCase):
    def test_android_client_posts_to_local_process_api(self) -> None:
        activity = ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "pocketcv" / "pdf" / "MainActivity.java"
        source = activity.read_text(encoding="utf-8")

        self.assertIn("http://10.0.2.2:8765", source)
        self.assertIn("OpenCVLoader.initLocal", source)
        self.assertIn("CornerOverlayView", source)
        self.assertIn("自動角に戻す", source)
        self.assertIn("端末内OpenCVでスキャン", source)
        self.assertIn("OnDeviceScanner.process", source)
        self.assertIn("PNG共有", source)
        self.assertIn("PDF共有", source)
        self.assertIn("DOCX共有", source)
        self.assertIn("FileProvider.getUriForFile", source)
        self.assertIn("Intent.ACTION_SEND", source)
        self.assertIn("manualCornersForSource", source)
        self.assertIn('"corners_space", "input"', source)
        self.assertIn("カメラで撮影", source)
        self.assertIn("MediaStore.ACTION_IMAGE_CAPTURE", source)
        self.assertIn("MediaStore.Images.Media.EXTERNAL_CONTENT_URI", source)
        self.assertIn("API確認", source)
        self.assertIn('normalizedEndpoint() + "/api/health"', source)
        self.assertIn('normalizedEndpoint() + "/api/process"', source)
        self.assertIn("multipart/form-data", source)
        self.assertIn("image_base64", source)
        self.assertIn("pdf_base64", source)
        self.assertNotIn("WebView", source)

    def test_android_app_bundles_opencv_for_on_device_scanning(self) -> None:
        build = (ROOT / "android" / "app" / "build.gradle").read_text(encoding="utf-8")
        gradle_properties = (ROOT / "android" / "gradle.properties").read_text(encoding="utf-8")
        scanner = (ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "pocketcv" / "pdf" / "OnDeviceScanner.java").read_text(encoding="utf-8")

        self.assertIn("android.useAndroidX=true", gradle_properties)
        self.assertIn('implementation "androidx.core:core:', build)
        self.assertIn('implementation "org.opencv:opencv:4.13.0"', build)
        self.assertIn("Imgproc.getPerspectiveTransform", scanner)
        self.assertIn("Imgproc.warpPerspective", scanner)
        self.assertIn("Imgproc.adaptiveThreshold", scanner)
        self.assertIn("detectCorners", scanner)
        self.assertIn("manual_overlay_homography", scanner)
        self.assertIn("PdfDocument", scanner)
        manifest = (ROOT / "android" / "app" / "src" / "main" / "AndroidManifest.xml").read_text(encoding="utf-8")
        file_paths = (ROOT / "android" / "app" / "src" / "main" / "res" / "xml" / "file_paths.xml").read_text(encoding="utf-8")

        self.assertIn("android.hardware.camera.any", manifest)
        self.assertIn("androidx.core.content.FileProvider", manifest)
        self.assertIn("${applicationId}.fileprovider", manifest)
        self.assertIn("cache-path", file_paths)
        self.assertIn("shared/", file_paths)

    def test_android_workflow_uploads_debug_apk(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "android.yml").read_text(encoding="utf-8")

        self.assertIn("Android APK", workflow)
        self.assertIn("working-directory: android", workflow)
        self.assertIn("./gradlew :app:assembleDebug", workflow)
        self.assertIn("pocketcv-android-debug-apk", workflow)
        self.assertIn("app-debug.apk", workflow)

    def test_android_gradle_wrapper_is_committed(self) -> None:
        wrapper_bat = ROOT / "android" / "gradlew.bat"
        wrapper_jar = ROOT / "android" / "gradle" / "wrapper" / "gradle-wrapper.jar"
        wrapper_props = ROOT / "android" / "gradle" / "wrapper" / "gradle-wrapper.properties"

        self.assertTrue(wrapper_bat.exists())
        self.assertTrue(wrapper_jar.exists())
        self.assertIn("gradle-8.10.2-bin.zip", wrapper_props.read_text(encoding="utf-8"))

    def test_local_backend_launcher_and_desktop_script_are_registered(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        launcher = (ROOT / "src" / "clearscan_cv" / "local_app.py").read_text(encoding="utf-8")
        build_script = (ROOT / "scripts" / "build_windows_local_app.ps1").read_text(encoding="utf-8")
        install_script = (ROOT / "scripts" / "install_android_debug.ps1").read_text(encoding="utf-8")
        emulator_qa_script = (ROOT / "scripts" / "android_emulator_qa.ps1").read_text(encoding="utf-8")

        self.assertIn('pocketcv-local = "clearscan_cv.local_app:main"', pyproject)
        self.assertIn("uvicorn.run", launcher)
        self.assertIn("/local", launcher)
        self.assertIn("PyInstaller", build_script)
        self.assertIn("adb", install_script)
        self.assertIn("com.pocketcv.pdf/.MainActivity", install_script)
        self.assertIn("PocketCV_API35", emulator_qa_script)
        self.assertIn("uiautomator dump", emulator_qa_script)
        self.assertIn("pocketcv-sample.jpg", emulator_qa_script)
        self.assertIn("カメラで撮影", emulator_qa_script)
        self.assertIn("自動角に戻す", emulator_qa_script)
        self.assertIn("四隅調整OK", emulator_qa_script)
        self.assertIn("端末内OpenCVでスキャン", emulator_qa_script)
        self.assertIn("PNG共有", emulator_qa_script)
        self.assertIn("PDF共有", emulator_qa_script)
        self.assertIn("PC後端でスキャン生成", emulator_qa_script)


if __name__ == "__main__":
    unittest.main()
