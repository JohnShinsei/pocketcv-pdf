package com.pocketcv.pdf;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import android.graphics.RectF;
import android.view.MotionEvent;
import android.view.View;

final class CornerOverlayView extends View {
    private static final int CORNER_COUNT = 4;
    private static final String[] LABELS = {"左上", "右上", "右下", "左下"};

    private final Paint backgroundPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint imageDimPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint fillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint linePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint handlePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint handleStrokePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint labelPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF imageRect = new RectF();
    private final PointF[] corners = new PointF[CORNER_COUNT];

    private Bitmap bitmap;
    private int draggingIndex = -1;

    CornerOverlayView(Context context) {
        super(context);
        setFocusable(true);
        setContentDescription("四隅調整フレーム");

        backgroundPaint.setColor(Color.rgb(18, 22, 26));
        imageDimPaint.setColor(Color.argb(65, 0, 0, 0));
        fillPaint.setColor(Color.argb(62, 20, 184, 166));
        linePaint.setColor(Color.rgb(20, 184, 166));
        linePaint.setStyle(Paint.Style.STROKE);
        linePaint.setStrokeCap(Paint.Cap.ROUND);
        linePaint.setStrokeJoin(Paint.Join.ROUND);
        handlePaint.setColor(Color.WHITE);
        handlePaint.setStyle(Paint.Style.FILL);
        handleStrokePaint.setColor(Color.rgb(20, 184, 166));
        handleStrokePaint.setStyle(Paint.Style.STROKE);
        labelPaint.setColor(Color.rgb(18, 22, 26));
        labelPaint.setTextAlign(Paint.Align.CENTER);
        labelPaint.setFakeBoldText(true);

        for (int i = 0; i < CORNER_COUNT; i++) {
            corners[i] = new PointF();
        }
    }

    void setBitmap(Bitmap bitmap) {
        this.bitmap = bitmap;
        resetCorners();
    }

    int getImageWidth() {
        return bitmap == null ? 0 : bitmap.getWidth();
    }

    int getImageHeight() {
        return bitmap == null ? 0 : bitmap.getHeight();
    }

    float[] getCorners() {
        if (bitmap == null) {
            return null;
        }
        float[] values = new float[CORNER_COUNT * 2];
        for (int i = 0; i < CORNER_COUNT; i++) {
            values[i * 2] = corners[i].x;
            values[i * 2 + 1] = corners[i].y;
        }
        return values;
    }

    void setCorners(float[] values) {
        if (bitmap == null || values == null || values.length < CORNER_COUNT * 2) {
            return;
        }
        for (int i = 0; i < CORNER_COUNT; i++) {
            corners[i].set(
                    clamp(values[i * 2], 0, bitmap.getWidth() - 1),
                    clamp(values[i * 2 + 1], 0, bitmap.getHeight() - 1)
            );
        }
        invalidate();
    }

    void resetCorners() {
        if (bitmap != null) {
            float marginX = Math.max(1f, bitmap.getWidth() * 0.03f);
            float marginY = Math.max(1f, bitmap.getHeight() * 0.03f);
            corners[0].set(marginX, marginY);
            corners[1].set(bitmap.getWidth() - marginX - 1, marginY);
            corners[2].set(bitmap.getWidth() - marginX - 1, bitmap.getHeight() - marginY - 1);
            corners[3].set(marginX, bitmap.getHeight() - marginY - 1);
        }
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawRect(0, 0, getWidth(), getHeight(), backgroundPaint);
        if (bitmap == null) {
            return;
        }

        updateImageRect();
        canvas.drawBitmap(bitmap, null, imageRect, null);
        drawOutsideDim(canvas);
        drawPolygon(canvas);
        drawHandles(canvas);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (bitmap == null) {
            return false;
        }
        updateImageRect();
        switch (event.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                draggingIndex = nearestHandle(event.getX(), event.getY());
                return draggingIndex >= 0;
            case MotionEvent.ACTION_MOVE:
                if (draggingIndex >= 0) {
                    PointF imagePoint = viewToImage(event.getX(), event.getY());
                    corners[draggingIndex].set(imagePoint.x, imagePoint.y);
                    invalidate();
                    return true;
                }
                return false;
            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_CANCEL:
                draggingIndex = -1;
                return true;
            default:
                return false;
        }
    }

    private void updateImageRect() {
        if (bitmap == null || getWidth() <= 0 || getHeight() <= 0) {
            imageRect.setEmpty();
            return;
        }
        float scale = Math.min((float) getWidth() / bitmap.getWidth(), (float) getHeight() / bitmap.getHeight());
        float width = bitmap.getWidth() * scale;
        float height = bitmap.getHeight() * scale;
        float left = (getWidth() - width) / 2f;
        float top = (getHeight() - height) / 2f;
        imageRect.set(left, top, left + width, top + height);
    }

    private void drawOutsideDim(Canvas canvas) {
        Path polygon = polygonPath();
        int checkpoint = canvas.save();
        canvas.clipOutPath(polygon);
        canvas.drawRect(imageRect, imageDimPaint);
        canvas.restoreToCount(checkpoint);
    }

    private void drawPolygon(Canvas canvas) {
        linePaint.setStrokeWidth(dp(2.5f));
        canvas.drawPath(polygonPath(), fillPaint);
        canvas.drawPath(polygonPath(), linePaint);
    }

    private void drawHandles(Canvas canvas) {
        float radius = dp(19);
        handleStrokePaint.setStrokeWidth(dp(3));
        labelPaint.setTextSize(dp(13));
        Paint.FontMetrics metrics = labelPaint.getFontMetrics();
        float textOffset = -(metrics.ascent + metrics.descent) / 2f;
        for (int i = 0; i < CORNER_COUNT; i++) {
            PointF viewPoint = imageToView(corners[i]);
            canvas.drawCircle(viewPoint.x, viewPoint.y, radius, handlePaint);
            canvas.drawCircle(viewPoint.x, viewPoint.y, radius, handleStrokePaint);
            canvas.drawText(LABELS[i], viewPoint.x, viewPoint.y + textOffset, labelPaint);
        }
    }

    private Path polygonPath() {
        Path path = new Path();
        PointF first = imageToView(corners[0]);
        path.moveTo(first.x, first.y);
        for (int i = 1; i < CORNER_COUNT; i++) {
            PointF point = imageToView(corners[i]);
            path.lineTo(point.x, point.y);
        }
        path.close();
        return path;
    }

    private int nearestHandle(float x, float y) {
        float maxDistance = dp(42);
        int nearest = -1;
        float best = Float.MAX_VALUE;
        for (int i = 0; i < CORNER_COUNT; i++) {
            PointF point = imageToView(corners[i]);
            float distance = (float) Math.hypot(x - point.x, y - point.y);
            if (distance <= maxDistance && distance < best) {
                nearest = i;
                best = distance;
            }
        }
        return nearest;
    }

    private PointF imageToView(PointF imagePoint) {
        if (bitmap == null || imageRect.isEmpty()) {
            return new PointF();
        }
        float scaleX = imageRect.width() / bitmap.getWidth();
        float scaleY = imageRect.height() / bitmap.getHeight();
        return new PointF(imageRect.left + imagePoint.x * scaleX, imageRect.top + imagePoint.y * scaleY);
    }

    private PointF viewToImage(float x, float y) {
        float imageX = (x - imageRect.left) * bitmap.getWidth() / Math.max(1f, imageRect.width());
        float imageY = (y - imageRect.top) * bitmap.getHeight() / Math.max(1f, imageRect.height());
        return new PointF(
                clamp(imageX, 0, bitmap.getWidth() - 1),
                clamp(imageY, 0, bitmap.getHeight() - 1)
        );
    }

    private float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    private float dp(float value) {
        return value * getResources().getDisplayMetrics().density;
    }
}
