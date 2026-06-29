package com.pocketcv.pdf;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.RectF;
import android.graphics.pdf.PdfDocument;

import org.json.JSONObject;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

final class OnDeviceScanner {
    private static final int MAX_INPUT_EDGE = 2400;
    private static final int MAX_OUTPUT_EDGE = 2200;

    static final class Result {
        final Bitmap previewBitmap;
        final byte[] imageBytes;
        final byte[] pdfBytes;
        final JSONObject report;

        Result(Bitmap previewBitmap, byte[] imageBytes, byte[] pdfBytes, JSONObject report) {
            this.previewBitmap = previewBitmap;
            this.imageBytes = imageBytes;
            this.pdfBytes = pdfBytes;
            this.report = report;
        }
    }

    private static final class Quad {
        final Point[] points;
        final String method;
        final double confidence;

        Quad(Point[] points, String method, double confidence) {
            this.points = points;
            this.method = method;
            this.confidence = confidence;
        }
    }

    private OnDeviceScanner() {
    }

    static float[] detectCorners(byte[] imageBytes, int targetWidth, int targetHeight) throws Exception {
        Bitmap input = decodeScaledBitmap(imageBytes);
        Mat rgba = new Mat();
        Mat bgr = new Mat();
        try {
            Utils.bitmapToMat(input, rgba);
            Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR);
            Quad quad = detectDocument(bgr);
            return scaleCorners(quad.points, bgr.cols(), bgr.rows(), targetWidth, targetHeight);
        } finally {
            rgba.release();
            bgr.release();
        }
    }

    static Result process(byte[] imageBytes, String filename, String mode) throws Exception {
        return process(imageBytes, filename, mode, null, 0, 0);
    }

    static Result process(
            byte[] imageBytes,
            String filename,
            String mode,
            float[] manualCorners,
            int cornerImageWidth,
            int cornerImageHeight
    ) throws Exception {
        Bitmap input = decodeScaledBitmap(imageBytes);
        Mat rgba = new Mat();
        Mat bgr = new Mat();
        Mat warped = new Mat();
        Mat output = new Mat();
        try {
            Utils.bitmapToMat(input, rgba);
            Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR);
            Quad quad = manualQuad(manualCorners, cornerImageWidth, cornerImageHeight, bgr.cols(), bgr.rows());
            if (quad == null) {
                quad = detectDocument(bgr);
            }
            Size outputSize = warpSize(quad.points);
            Point[] target = new Point[]{
                    new Point(0, 0),
                    new Point(outputSize.width - 1, 0),
                    new Point(outputSize.width - 1, outputSize.height - 1),
                    new Point(0, outputSize.height - 1)
            };
            Mat transform = Imgproc.getPerspectiveTransform(
                    new MatOfPoint2f(quad.points),
                    new MatOfPoint2f(target)
            );
            try {
                Imgproc.warpPerspective(bgr, warped, transform, outputSize, Imgproc.INTER_CUBIC);
            } finally {
                transform.release();
            }

            output = enhance(warped, mode);
            Bitmap preview = matToBitmap(output);
            byte[] png = encodePng(preview);
            byte[] pdf = buildPdf(preview, filename);
            JSONObject report = new JSONObject();
            report.put("engine", "android-opencv");
            report.put("offline_processing", true);
            report.put("mode", normalizedMode(mode));
            report.put("input_width", bgr.cols());
            report.put("input_height", bgr.rows());
            report.put("output_width", output.cols());
            report.put("output_height", output.rows());
            JSONObject detection = new JSONObject();
            detection.put("method", quad.method);
            detection.put("confidence", quad.confidence);
            detection.put("corners", cornersJson(quad.points));
            report.put("document_detection", detection);
            report.put("manual_corners", "manual_overlay_homography".equals(quad.method));
            report.put("exports", "png,pdf");
            return new Result(preview, png, pdf, report);
        } finally {
            rgba.release();
            bgr.release();
            warped.release();
            output.release();
        }
    }

    private static Quad manualQuad(float[] corners, int sourceWidth, int sourceHeight, int targetWidth, int targetHeight) {
        if (corners == null || corners.length < 8 || sourceWidth <= 0 || sourceHeight <= 0) {
            return null;
        }
        Point[] scaled = new Point[4];
        double scaleX = (double) targetWidth / Math.max(1, sourceWidth);
        double scaleY = (double) targetHeight / Math.max(1, sourceHeight);
        for (int i = 0; i < 4; i++) {
            double x = clamp(corners[i * 2] * scaleX, 0, targetWidth - 1);
            double y = clamp(corners[i * 2 + 1] * scaleY, 0, targetHeight - 1);
            scaled[i] = new Point(x, y);
        }
        Point[] ordered = order(scaled);
        double area = Math.abs(polygonArea(ordered));
        if (area < Math.max(1.0, targetWidth * targetHeight * 0.01)) {
            return null;
        }
        return new Quad(ordered, "manual_overlay_homography", 1.0);
    }

    private static Bitmap decodeScaledBitmap(byte[] imageBytes) {
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length, bounds);
        int sample = 1;
        int maxEdge = Math.max(bounds.outWidth, bounds.outHeight);
        while (maxEdge / sample > MAX_INPUT_EDGE) {
            sample *= 2;
        }
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = sample;
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length, options);
    }

    private static Quad detectDocument(Mat bgr) {
        double scale = Math.min(1.0, 1100.0 / Math.max(bgr.cols(), bgr.rows()));
        Mat work = new Mat();
        Mat gray = new Mat();
        Mat edges = new Mat();
        Mat closed = new Mat();
        Mat hierarchy = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();
        try {
            if (scale < 1.0) {
                Imgproc.resize(bgr, work, new Size(bgr.cols() * scale, bgr.rows() * scale));
            } else {
                work = bgr.clone();
            }
            Imgproc.cvtColor(work, gray, Imgproc.COLOR_BGR2GRAY);
            Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 0);
            Imgproc.Canny(gray, edges, 55, 150);
            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
            try {
                Imgproc.morphologyEx(edges, closed, Imgproc.MORPH_CLOSE, kernel);
                Imgproc.dilate(closed, closed, kernel);
            } finally {
                kernel.release();
            }
            Imgproc.findContours(closed, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            double imageArea = work.cols() * work.rows();
            Point[] best = null;
            double bestArea = 0.0;
            for (MatOfPoint contour : contours) {
                double area = Math.abs(Imgproc.contourArea(contour));
                if (area < imageArea * 0.06) {
                    continue;
                }
                MatOfPoint2f curve = new MatOfPoint2f(contour.toArray());
                try {
                    double peri = Imgproc.arcLength(curve, true);
                    for (double epsilonFactor : new double[]{0.018, 0.025, 0.035, 0.05}) {
                        MatOfPoint2f approx = new MatOfPoint2f();
                        try {
                            Imgproc.approxPolyDP(curve, approx, peri * epsilonFactor, true);
                            if (approx.total() == 4) {
                                Point[] points = approx.toArray();
                                MatOfPoint polygon = new MatOfPoint(points);
                                try {
                                    if (Imgproc.isContourConvex(polygon) && area > bestArea) {
                                        best = order(points);
                                        bestArea = area;
                                    }
                                } finally {
                                    polygon.release();
                                }
                            }
                        } finally {
                            approx.release();
                        }
                    }
                } finally {
                    curve.release();
                }
            }
            if (best != null) {
                Point[] original = new Point[4];
                for (int i = 0; i < 4; i++) {
                    original[i] = new Point(best[i].x / scale, best[i].y / scale);
                }
                double confidence = Math.min(1.0, Math.max(0.35, bestArea / Math.max(1.0, imageArea)));
                return new Quad(original, "contour_canny_homography", confidence);
            }
            Point[] full = new Point[]{
                    new Point(0, 0),
                    new Point(bgr.cols() - 1, 0),
                    new Point(bgr.cols() - 1, bgr.rows() - 1),
                    new Point(0, bgr.rows() - 1)
            };
            return new Quad(full, "full_image_fallback", 0.2);
        } finally {
            for (MatOfPoint contour : contours) {
                contour.release();
            }
            work.release();
            gray.release();
            edges.release();
            closed.release();
            hierarchy.release();
        }
    }

    private static Point[] order(Point[] input) {
        Point[] points = input.clone();
        Arrays.sort(points, Comparator.comparingDouble(p -> p.y));
        Point[] top = new Point[]{points[0], points[1]};
        Point[] bottom = new Point[]{points[2], points[3]};
        Arrays.sort(top, Comparator.comparingDouble(p -> p.x));
        Arrays.sort(bottom, Comparator.comparingDouble(p -> p.x));
        return new Point[]{top[0], top[1], bottom[1], bottom[0]};
    }

    private static Size warpSize(Point[] quad) {
        double top = distance(quad[0], quad[1]);
        double bottom = distance(quad[3], quad[2]);
        double left = distance(quad[0], quad[3]);
        double right = distance(quad[1], quad[2]);
        double width = Math.max(top, bottom);
        double height = Math.max(left, right);
        double scale = Math.min(1.0, MAX_OUTPUT_EDGE / Math.max(width, height));
        return new Size(Math.max(64, Math.round(width * scale)), Math.max(64, Math.round(height * scale)));
    }

    private static double distance(Point a, Point b) {
        return Math.hypot(a.x - b.x, a.y - b.y);
    }

    private static double polygonArea(Point[] points) {
        double area = 0.0;
        for (int i = 0; i < points.length; i++) {
            Point current = points[i];
            Point next = points[(i + 1) % points.length];
            area += current.x * next.y - next.x * current.y;
        }
        return area / 2.0;
    }

    private static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    private static float[] scaleCorners(Point[] points, int sourceWidth, int sourceHeight, int targetWidth, int targetHeight) {
        float[] scaled = new float[8];
        double scaleX = (double) Math.max(1, targetWidth) / Math.max(1, sourceWidth);
        double scaleY = (double) Math.max(1, targetHeight) / Math.max(1, sourceHeight);
        for (int i = 0; i < points.length; i++) {
            scaled[i * 2] = (float) clamp(points[i].x * scaleX, 0, Math.max(0, targetWidth - 1));
            scaled[i * 2 + 1] = (float) clamp(points[i].y * scaleY, 0, Math.max(0, targetHeight - 1));
        }
        return scaled;
    }

    private static Mat enhance(Mat warped, String mode) {
        String normalized = normalizedMode(mode);
        if ("color".equals(normalized)) {
            return warped.clone();
        }
        Mat gray = new Mat();
        Mat normalizedGray = new Mat();
        try {
            Imgproc.cvtColor(warped, gray, Imgproc.COLOR_BGR2GRAY);
            normalizeIllumination(gray, normalizedGray);
            if ("binary".equals(normalized)) {
                Mat binary = new Mat();
                Imgproc.adaptiveThreshold(
                        normalizedGray,
                        binary,
                        255,
                        Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                        Imgproc.THRESH_BINARY,
                        adaptiveBlockSize(normalizedGray),
                        9
                );
                Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2));
                try {
                    Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, kernel);
                } finally {
                    kernel.release();
                }
                return binary;
            }
            return normalizedGray.clone();
        } finally {
            gray.release();
            normalizedGray.release();
        }
    }

    private static void normalizeIllumination(Mat gray, Mat output) {
        Mat background = new Mat();
        Mat grayFloat = new Mat();
        Mat backgroundFloat = new Mat();
        Mat normalizedFloat = new Mat();
        try {
            double sigma = Math.max(18.0, Math.min(gray.cols(), gray.rows()) / 28.0);
            Imgproc.GaussianBlur(gray, background, new Size(0, 0), sigma);
            gray.convertTo(grayFloat, CvType.CV_32F);
            background.convertTo(backgroundFloat, CvType.CV_32F);
            Core.add(backgroundFloat, new Scalar(1.0), backgroundFloat);
            Core.divide(grayFloat, backgroundFloat, normalizedFloat, 245.0);
            Core.normalize(normalizedFloat, normalizedFloat, 0, 255, Core.NORM_MINMAX);
            normalizedFloat.convertTo(output, CvType.CV_8U);
            CLAHE clahe = Imgproc.createCLAHE(1.4, new Size(8, 8));
            clahe.apply(output, output);
        } finally {
            background.release();
            grayFloat.release();
            backgroundFloat.release();
            normalizedFloat.release();
        }
    }

    private static int adaptiveBlockSize(Mat image) {
        int block = Math.max(31, Math.min(image.cols(), image.rows()) / 22);
        return block % 2 == 0 ? block + 1 : block;
    }

    private static String normalizedMode(String mode) {
        if ("binary".equals(mode) || "gray".equals(mode) || "color".equals(mode)) {
            return mode;
        }
        return "gray";
    }

    private static Bitmap matToBitmap(Mat mat) {
        Mat rgba = new Mat();
        try {
            if (mat.channels() == 1) {
                Imgproc.cvtColor(mat, rgba, Imgproc.COLOR_GRAY2RGBA);
            } else {
                Imgproc.cvtColor(mat, rgba, Imgproc.COLOR_BGR2RGBA);
            }
            Bitmap bitmap = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rgba, bitmap);
            return bitmap;
        } finally {
            rgba.release();
        }
    }

    private static byte[] encodePng(Bitmap bitmap) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
        return out.toByteArray();
    }

    private static byte[] buildPdf(Bitmap bitmap, String filename) throws Exception {
        PdfDocument document = new PdfDocument();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try {
            PdfDocument.PageInfo pageInfo = new PdfDocument.PageInfo.Builder(1240, 1754, 1).create();
            PdfDocument.Page page = document.startPage(pageInfo);
            page.getCanvas().drawColor(Color.WHITE);
            RectF target = fit(bitmap.getWidth(), bitmap.getHeight(), pageInfo.getPageWidth(), pageInfo.getPageHeight());
            page.getCanvas().drawBitmap(bitmap, null, target, null);
            document.finishPage(page);
            document.writeTo(out);
            return out.toByteArray();
        } finally {
            document.close();
        }
    }

    private static RectF fit(int imageWidth, int imageHeight, int pageWidth, int pageHeight) {
        float margin = 48f;
        float usableWidth = pageWidth - margin * 2f;
        float usableHeight = pageHeight - margin * 2f;
        float scale = Math.min(usableWidth / imageWidth, usableHeight / imageHeight);
        float width = imageWidth * scale;
        float height = imageHeight * scale;
        float left = (pageWidth - width) / 2f;
        float top = (pageHeight - height) / 2f;
        return new RectF(left, top, left + width, top + height);
    }

    private static String cornersJson(Point[] points) {
        StringBuilder builder = new StringBuilder("[");
        for (int i = 0; i < points.length; i++) {
            if (i > 0) {
                builder.append(',');
            }
            builder.append(String.format(Locale.ROOT, "[%.1f,%.1f]", points[i].x, points[i].y));
        }
        builder.append(']');
        return builder.toString();
    }
}
