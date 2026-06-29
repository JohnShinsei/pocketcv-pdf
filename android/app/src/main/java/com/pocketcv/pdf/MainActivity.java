package com.pocketcv.pdf;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.provider.OpenableColumns;
import android.text.InputType;
import android.util.Base64;
import android.view.Gravity;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.core.content.FileProvider;

import org.json.JSONObject;
import org.opencv.android.OpenCVLoader;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

public class MainActivity extends Activity {
    private static final int REQ_PICK_IMAGE = 1001;
    private static final int REQ_SAVE_IMAGE = 1002;
    private static final int REQ_SAVE_PDF = 1003;
    private static final int REQ_SAVE_DOCX = 1004;
    private static final int REQ_CAPTURE_IMAGE = 1005;

    private EditText endpointInput;
    private Spinner modeSpinner;
    private CheckBox readabilityCheck;
    private CheckBox ocrCheck;
    private CheckBox searchablePdfCheck;
    private CheckBox docxCheck;
    private TextView statusText;
    private TextView reportText;
    private ImageView previewImage;
    private Button healthButton;
    private Button cameraButton;
    private Button onDeviceProcessButton;
    private Button processButton;
    private Button saveImageButton;
    private Button savePdfButton;
    private Button saveDocxButton;
    private boolean opencvReady;
    private String opencvStatus = "";
    private boolean busy;

    private Uri selectedImageUri;
    private Uri pendingCameraUri;
    private String selectedImageName = "scan.jpg";
    private String pendingCameraName = "capture.jpg";
    private byte[] latestImageBytes;
    private byte[] latestPdfBytes;
    private byte[] latestDocxBytes;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        initializeOpenCv();
        setContentView(buildUi());
    }

    private View buildUi() {
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(28, 28, 28, 36);
        scroll.addView(root);

        TextView title = new TextView(this);
        title.setText("PocketCV PDF Android");
        title.setTextSize(24);
        title.setGravity(Gravity.START);
        root.addView(title);

        TextView intro = new TextView(this);
        intro.setText("Android で画像を選択し、端末内 OpenCV または PC/LAN 内の FastAPI 後端でスキャン画像を生成します。エミュレーターから PC 後端へ接続する場合は 10.0.2.2 を使用します。");
        intro.setPadding(0, 8, 0, 20);
        root.addView(intro);

        endpointInput = new EditText(this);
        endpointInput.setSingleLine(true);
        endpointInput.setInputType(InputType.TYPE_TEXT_VARIATION_URI);
        endpointInput.setText("http://10.0.2.2:8765");
        endpointInput.setHint("Backend URL");
        root.addView(endpointInput, matchWidth());

        healthButton = new Button(this);
        healthButton.setText("API確認");
        healthButton.setOnClickListener(v -> checkBackend());
        root.addView(healthButton, matchWidth());

        modeSpinner = new Spinner(this);
        ArrayAdapter<String> modes = new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, new String[]{
                "auto", "gray", "binary", "color"
        });
        modeSpinner.setAdapter(modes);
        root.addView(modeSpinner, matchWidth());

        readabilityCheck = checkbox("品質診断", true);
        ocrCheck = checkbox("OCR 実行", false);
        searchablePdfCheck = checkbox("OCR 文字層付き PDF", false);
        docxCheck = checkbox("DOCX 生成", false);
        root.addView(readabilityCheck);
        root.addView(ocrCheck);
        root.addView(searchablePdfCheck);
        root.addView(docxCheck);

        searchablePdfCheck.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                ocrCheck.setChecked(true);
            }
        });
        docxCheck.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                ocrCheck.setChecked(true);
            }
        });

        cameraButton = new Button(this);
        cameraButton.setText("カメラで撮影");
        cameraButton.setOnClickListener(v -> captureImage());
        root.addView(cameraButton, matchWidth());

        Button pickButton = new Button(this);
        pickButton.setText("画像を選択");
        pickButton.setOnClickListener(v -> pickImage());
        root.addView(pickButton, matchWidth());

        onDeviceProcessButton = new Button(this);
        onDeviceProcessButton.setText("端末内OpenCVでスキャン");
        onDeviceProcessButton.setEnabled(false);
        onDeviceProcessButton.setOnClickListener(v -> processImageOnDevice());
        root.addView(onDeviceProcessButton, matchWidth());

        processButton = new Button(this);
        processButton.setText("PC後端でスキャン生成");
        processButton.setEnabled(false);
        processButton.setOnClickListener(v -> processImage());
        root.addView(processButton, matchWidth());

        LinearLayout saveRow = new LinearLayout(this);
        saveRow.setOrientation(LinearLayout.HORIZONTAL);
        saveImageButton = saveButton("PNG保存", REQ_SAVE_IMAGE);
        savePdfButton = saveButton("PDF保存", REQ_SAVE_PDF);
        saveDocxButton = saveButton("DOCX保存", REQ_SAVE_DOCX);
        saveRow.addView(saveImageButton, rowWeight());
        saveRow.addView(savePdfButton, rowWeight());
        saveRow.addView(saveDocxButton, rowWeight());
        root.addView(saveRow, matchWidth());

        statusText = new TextView(this);
        statusText.setText(opencvReady ? "画像待ち · OpenCV準備OK" : "画像待ち · OpenCV未初期化");
        statusText.setPadding(0, 10, 0, 10);
        root.addView(statusText);

        previewImage = new ImageView(this);
        previewImage.setAdjustViewBounds(true);
        previewImage.setScaleType(ImageView.ScaleType.FIT_CENTER);
        previewImage.setPadding(0, 18, 0, 18);
        root.addView(previewImage, matchWidth());

        reportText = new TextView(this);
        reportText.setTextIsSelectable(true);
        reportText.setText("");
        root.addView(reportText);

        modeSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        return scroll;
    }

    private void initializeOpenCv() {
        try {
            opencvReady = OpenCVLoader.initLocal();
            opencvStatus = opencvReady ? "OpenCV ready" : "OpenCV initLocal returned false";
        } catch (Throwable error) {
            try {
                System.loadLibrary("opencv_java4");
                opencvReady = true;
                opencvStatus = "OpenCV ready via System.loadLibrary";
            } catch (Throwable fallbackError) {
                opencvReady = false;
                opencvStatus = fallbackError.getMessage();
            }
        }
    }

    private LinearLayout.LayoutParams matchWidth() {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        params.setMargins(0, 8, 0, 8);
        return params;
    }

    private LinearLayout.LayoutParams rowWeight() {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1);
        params.setMargins(4, 8, 4, 8);
        return params;
    }

    private CheckBox checkbox(String label, boolean checked) {
        CheckBox box = new CheckBox(this);
        box.setText(label);
        box.setChecked(checked);
        return box;
    }

    private Button saveButton(String label, int requestCode) {
        Button button = new Button(this);
        button.setText(label);
        button.setEnabled(false);
        button.setOnClickListener(v -> saveLatest(requestCode));
        return button;
    }

    private void pickImage() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        startActivityForResult(intent, REQ_PICK_IMAGE);
    }

    private void captureImage() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (intent.resolveActivity(getPackageManager()) == null) {
            setStatus("カメラアプリが見つかりません。");
            return;
        }
        try {
            pendingCameraName = "pocketcv-capture-" + System.currentTimeMillis() + ".jpg";
            File photoFile = new File(getCacheDir(), pendingCameraName);
            pendingCameraUri = FileProvider.getUriForFile(
                    this,
                    getPackageName() + ".fileprovider",
                    photoFile
            );
            intent.putExtra(MediaStore.EXTRA_OUTPUT, pendingCameraUri);
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
            startActivityForResult(intent, REQ_CAPTURE_IMAGE);
        } catch (Exception error) {
            setStatus("カメラ起動失敗: " + error.getMessage());
        }
    }

    private void processImage() {
        if (selectedImageUri == null) {
            setStatus("画像を選択してください。");
            return;
        }
        busy = true;
        processButton.setEnabled(false);
        setStatus("ローカル Python/OpenCV 後端で処理中...");
        reportText.setText("");
        latestImageBytes = null;
        latestPdfBytes = null;
        latestDocxBytes = null;
        updateSaveButtons();

        new Thread(() -> {
            try {
                byte[] inputBytes = readAllBytes(selectedImageUri);
                JSONObject payload = postProcess(inputBytes);
                String imageBase64 = payload.optString("image_base64", "");
                if (!imageBase64.isEmpty()) {
                    latestImageBytes = Base64.decode(imageBase64, Base64.DEFAULT);
                }
                String pdfBase64 = payload.optString("pdf_base64", "");
                if (!pdfBase64.isEmpty()) {
                    latestPdfBytes = Base64.decode(pdfBase64, Base64.DEFAULT);
                }
                String docxBase64 = payload.optString("docx_base64", "");
                if (!docxBase64.isEmpty()) {
                    latestDocxBytes = Base64.decode(docxBase64, Base64.DEFAULT);
                }
                String prettyPayload = payload.toString(2);

                runOnUiThread(() -> {
                    if (latestImageBytes != null) {
                        Bitmap bitmap = BitmapFactory.decodeByteArray(latestImageBytes, 0, latestImageBytes.length);
                        previewImage.setImageBitmap(bitmap);
                    }
                    reportText.setText(prettyPayload);
                    busy = false;
                    updateSaveButtons();
                    setStatus("完了");
                });
            } catch (Exception error) {
                runOnUiThread(() -> {
                    busy = false;
                    setStatus("失敗: " + error.getMessage());
                    updateSaveButtons();
                });
            }
        }).start();
    }

    private void processImageOnDevice() {
        if (selectedImageUri == null) {
            setStatus("画像を選択してください。");
            return;
        }
        if (!opencvReady) {
            setStatus("OpenCVを初期化できません: " + opencvStatus);
            return;
        }
        busy = true;
        onDeviceProcessButton.setEnabled(false);
        processButton.setEnabled(false);
        setStatus("端末内 OpenCV で処理中...");
        reportText.setText("");
        latestImageBytes = null;
        latestPdfBytes = null;
        latestDocxBytes = null;
        updateSaveButtons();

        new Thread(() -> {
            try {
                byte[] inputBytes = readAllBytes(selectedImageUri);
                OnDeviceScanner.Result result = OnDeviceScanner.process(inputBytes, selectedImageName, selectedMode());
                latestImageBytes = result.imageBytes;
                latestPdfBytes = result.pdfBytes;
                latestDocxBytes = null;
                String prettyReport = result.report.toString(2);
                runOnUiThread(() -> {
                    previewImage.setImageBitmap(result.previewBitmap);
                    reportText.setText(prettyReport);
                    busy = false;
                    updateSaveButtons();
                    setStatus("端末内処理完了");
                });
            } catch (Exception error) {
                runOnUiThread(() -> {
                    busy = false;
                    setStatus("端末内処理失敗: " + error.getMessage());
                    updateSaveButtons();
                });
            }
        }).start();
    }

    private void checkBackend() {
        healthButton.setEnabled(false);
        setStatus("API確認中...");
        new Thread(() -> {
            try {
                URL url = new URL(normalizedEndpoint() + "/api/health");
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setConnectTimeout(8000);
                connection.setReadTimeout(12000);
                connection.setRequestMethod("GET");
                int code = connection.getResponseCode();
                byte[] responseBytes;
                try (InputStream in = code >= 200 && code < 300 ? connection.getInputStream() : connection.getErrorStream()) {
                    responseBytes = readAllBytes(in);
                }
                String response = new String(responseBytes, StandardCharsets.UTF_8);
                if (code < 200 || code >= 300) {
                    throw new IllegalStateException(response);
                }
                JSONObject payload = new JSONObject(response);
                String engine = payload.optString("recommended_ocr_engine", "未導入");
                runOnUiThread(() -> {
                    reportText.setText(payload.toString());
                    setStatus("API OK · OCR " + engine);
                    healthButton.setEnabled(true);
                });
            } catch (Exception error) {
                runOnUiThread(() -> {
                    setStatus("API確認失敗: " + error.getMessage());
                    healthButton.setEnabled(true);
                });
            }
        }).start();
    }

    private JSONObject postProcess(byte[] imageBytes) throws Exception {
        URL url = new URL(normalizedEndpoint() + "/api/process");
        String boundary = "PocketCV" + System.currentTimeMillis();
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setConnectTimeout(15000);
        connection.setReadTimeout(180000);
        connection.setRequestMethod("POST");
        connection.setDoOutput(true);
        connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);
        try (OutputStream out = connection.getOutputStream()) {
            writeField(out, boundary, "mode", selectedMode());
            writeField(out, boundary, "auto_warp", "true");
            writeField(out, boundary, "auto_dewarp", "true");
            writeField(out, boundary, "readability", readabilityCheck.isChecked() ? "true" : "false");
            writeField(out, boundary, "ocr", ocrCheck.isChecked() ? "true" : "false");
            writeField(out, boundary, "ocr_lang", "jpn+eng");
            writeField(out, boundary, "searchable_pdf", searchablePdfCheck.isChecked() ? "true" : "false");
            writeField(out, boundary, "docx", docxCheck.isChecked() ? "true" : "false");
            writeField(out, boundary, "pdf", "true");
            writeFile(out, boundary, "file", selectedImageName, imageBytes);
            out.write(("--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8));
        }

        int code = connection.getResponseCode();
        byte[] responseBytes;
        try (InputStream in = code >= 200 && code < 300 ? connection.getInputStream() : connection.getErrorStream()) {
            responseBytes = readAllBytes(in);
        }
        String response = new String(responseBytes, StandardCharsets.UTF_8);
        if (code < 200 || code >= 300) {
            throw new IllegalStateException(response);
        }
        return new JSONObject(response);
    }

    private String normalizedEndpoint() {
        String endpoint = endpointInput.getText().toString().trim();
        while (endpoint.endsWith("/")) {
            endpoint = endpoint.substring(0, endpoint.length() - 1);
        }
        return endpoint;
    }

    private String selectedMode() {
        Object value = modeSpinner.getSelectedItem();
        return value == null ? "auto" : value.toString();
    }

    private void writeField(OutputStream out, String boundary, String name, String value) throws Exception {
        out.write(("--" + boundary + "\r\n").getBytes(StandardCharsets.UTF_8));
        out.write(("Content-Disposition: form-data; name=\"" + name + "\"\r\n\r\n").getBytes(StandardCharsets.UTF_8));
        out.write((value + "\r\n").getBytes(StandardCharsets.UTF_8));
    }

    private void writeFile(OutputStream out, String boundary, String name, String filename, byte[] bytes) throws Exception {
        out.write(("--" + boundary + "\r\n").getBytes(StandardCharsets.UTF_8));
        out.write(("Content-Disposition: form-data; name=\"" + name + "\"; filename=\"" + filename + "\"\r\n").getBytes(StandardCharsets.UTF_8));
        out.write("Content-Type: image/jpeg\r\n\r\n".getBytes(StandardCharsets.UTF_8));
        out.write(bytes);
        out.write("\r\n".getBytes(StandardCharsets.UTF_8));
    }

    private byte[] readAllBytes(Uri uri) throws Exception {
        try (InputStream in = getContentResolver().openInputStream(uri)) {
            return readAllBytes(in);
        }
    }

    private byte[] readAllBytes(InputStream in) throws Exception {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        byte[] chunk = new byte[8192];
        int read;
        while ((read = in.read(chunk)) != -1) {
            buffer.write(chunk, 0, read);
        }
        return buffer.toByteArray();
    }

    private String displayName(Uri uri) {
        try (android.database.Cursor cursor = getContentResolver().query(uri, null, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (index >= 0) {
                    return cursor.getString(index);
                }
            }
        } catch (Exception ignored) {
        }
        return "scan.jpg";
    }

    private void saveLatest(int requestCode) {
        byte[] bytes = bytesForRequest(requestCode);
        if (bytes == null) {
            return;
        }
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        if (requestCode == REQ_SAVE_PDF) {
            intent.setType("application/pdf");
            intent.putExtra(Intent.EXTRA_TITLE, outputName(".pdf"));
        } else if (requestCode == REQ_SAVE_DOCX) {
            intent.setType("application/vnd.openxmlformats-officedocument.wordprocessingml.document");
            intent.putExtra(Intent.EXTRA_TITLE, outputName(".docx"));
        } else {
            intent.setType("image/png");
            intent.putExtra(Intent.EXTRA_TITLE, outputName("-scan.png"));
        }
        startActivityForResult(intent, requestCode);
    }

    private byte[] bytesForRequest(int requestCode) {
        if (requestCode == REQ_SAVE_PDF) {
            return latestPdfBytes;
        }
        if (requestCode == REQ_SAVE_DOCX) {
            return latestDocxBytes;
        }
        return latestImageBytes;
    }

    private String outputName(String suffix) {
        String base = selectedImageName.replaceFirst("\\.[^.]+$", "");
        return String.format(Locale.ROOT, "%s%s", base, suffix);
    }

    private void updateSaveButtons() {
        saveImageButton.setEnabled(latestImageBytes != null);
        savePdfButton.setEnabled(latestPdfBytes != null);
        saveDocxButton.setEnabled(latestDocxBytes != null);
        if (processButton != null) {
            processButton.setEnabled(selectedImageUri != null && !busy);
        }
        if (onDeviceProcessButton != null) {
            onDeviceProcessButton.setEnabled(selectedImageUri != null && opencvReady && !busy);
        }
    }

    private void setStatus(String message) {
        statusText.setText(message);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQ_CAPTURE_IMAGE) {
            if (resultCode == RESULT_OK && pendingCameraUri != null) {
                setSelectedImage(pendingCameraUri, pendingCameraName);
            }
            return;
        }
        if (resultCode != RESULT_OK || data == null || data.getData() == null) {
            return;
        }
        Uri uri = data.getData();
        if (requestCode == REQ_PICK_IMAGE) {
            setSelectedImage(uri, displayName(uri));
            return;
        }

        byte[] bytes = bytesForRequest(requestCode);
        if (bytes == null) {
            return;
        }
        try (OutputStream out = getContentResolver().openOutputStream(uri)) {
            out.write(bytes);
            setStatus("保存しました");
        } catch (Exception error) {
            setStatus("保存失敗: " + error.getMessage());
        }
    }

    private void setSelectedImage(Uri uri, String name) {
        selectedImageUri = uri;
        selectedImageName = name;
        try (InputStream in = getContentResolver().openInputStream(uri)) {
            Bitmap bitmap = BitmapFactory.decodeStream(in);
            previewImage.setImageBitmap(bitmap);
            latestImageBytes = null;
            latestPdfBytes = null;
            latestDocxBytes = null;
            reportText.setText("");
            setStatus("選択済み: " + selectedImageName);
            updateSaveButtons();
        } catch (Exception error) {
            setStatus("画像を開けません: " + error.getMessage());
        }
    }
}
