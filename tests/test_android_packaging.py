from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AndroidPackagingTest(unittest.TestCase):
    def test_android_client_posts_to_local_process_api(self) -> None:
        activity = ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "pocketcv" / "pdf" / "MainActivity.java"
        source = activity.read_text(encoding="utf-8")

        self.assertIn("http://10.0.2.2:8765", source)
        self.assertIn("API確認", source)
        self.assertIn('normalizedEndpoint() + "/api/health"', source)
        self.assertIn('normalizedEndpoint() + "/api/process"', source)
        self.assertIn("multipart/form-data", source)
        self.assertIn("image_base64", source)
        self.assertIn("pdf_base64", source)
        self.assertNotIn("WebView", source)

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

        self.assertIn('pocketcv-local = "clearscan_cv.local_app:main"', pyproject)
        self.assertIn("uvicorn.run", launcher)
        self.assertIn("/local", launcher)
        self.assertIn("PyInstaller", build_script)
        self.assertIn("adb", install_script)
        self.assertIn("com.pocketcv.pdf/.MainActivity", install_script)


if __name__ == "__main__":
    unittest.main()
