package com.pocketcv.pdf;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
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

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
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
    private Button processButton;
    private Button saveImageButton;
    private Button savePdfButton;
    private Button saveDocxButton;

    private Uri selectedImageUri;
    private String selectedImageName = "scan.jpg";
    private byte[] latestImageBytes;
    private byte[] latestPdfBytes;
    private byte[] latestDocxBytes;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(buildUi());
    }

    private View buildUi() {
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(28, 28, 28, 36);
        scroll.addView(root);

        TextView title = new TextView(this);
        title.setText("PocketCV PDF Local");
        title.setTextSize(24);
        title.setGravity(Gravity.START);
        root.addView(title);

        TextView intro = new TextView(this);
        intro.setText("Android で画像を選択し、PC または LAN 内の FastAPI 後端へ送信します。Python/OpenCV がスキャン画像を生成します。エミュレーターでは 10.0.2.2 を使用します。");
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

        Button pickButton = new Button(this);
        pickButton.setText("画像を選択");
        pickButton.setOnClickListener(v -> pickImage());
        root.addView(pickButton, matchWidth());

        processButton = new Button(this);
        processButton.setText("ローカル後端でスキャン生成");
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

        previewImage = new ImageView(this);
        previewImage.setAdjustViewBounds(true);
        previewImage.setScaleType(ImageView.ScaleType.FIT_CENTER);
        previewImage.setPadding(0, 18, 0, 18);
        root.addView(previewImage, matchWidth());

        statusText = new TextView(this);
        statusText.setText("画像待ち");
        statusText.setPadding(0, 10, 0, 10);
        root.addView(statusText);

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

    private void processImage() {
        if (selectedImageUri == null) {
            setStatus("画像を選択してください。");
            return;
        }
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
                    updateSaveButtons();
                    setStatus("完了");
                });
            } catch (Exception error) {
                runOnUiThread(() -> {
                    setStatus("失敗: " + error.getMessage());
                    processButton.setEnabled(true);
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
        processButton.setEnabled(selectedImageUri != null);
    }

    private void setStatus(String message) {
        statusText.setText(message);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null || data.getData() == null) {
            return;
        }
        Uri uri = data.getData();
        if (requestCode == REQ_PICK_IMAGE) {
            selectedImageUri = uri;
            selectedImageName = displayName(uri);
            try {
                Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(uri));
                previewImage.setImageBitmap(bitmap);
                setStatus("選択済み: " + selectedImageName);
                processButton.setEnabled(true);
            } catch (Exception error) {
                setStatus("画像を開けません: " + error.getMessage());
            }
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
}
