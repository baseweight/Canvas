// SAM3 Image Processor for Baseweight Canvas
// Handles image preprocessing, coordinate transformation, and mask postprocessing

use anyhow::Result;
use image::{DynamicImage, GrayImage, RgbaImage, imageops::FilterType};
use ndarray::Array4;
use std::io::Cursor;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

// ImageNet normalization values (used by SAM3)
const SAM3_MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const SAM3_STD: [f32; 3] = [58.395, 57.12, 57.375];

#[derive(Debug, Clone)]
pub struct Sam3ImageProcessor {
    pub image_size: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for Sam3ImageProcessor {
    fn default() -> Self {
        Self {
            image_size: 1008,
            mean: SAM3_MEAN,
            std: SAM3_STD,
        }
    }
}

impl Sam3ImageProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Preprocess an image for SAM3 vision encoder
    /// Returns: (pixel_values, original_size, resized_size)
    /// pixel_values shape: [1, 3, image_size, image_size]
    pub fn preprocess(&self, image: DynamicImage) -> Result<(Array4<f32>, (u32, u32), (u32, u32))> {
        let original_size = (image.width(), image.height());

        // Convert to RGB
        let image = image.to_rgb8();

        // Resize to target size (1008x1008)
        let resized = image::imageops::resize(
            &image,
            self.image_size,
            self.image_size,
            FilterType::Lanczos3,
        );
        let resized_size = (self.image_size, self.image_size);

        // Create array with shape [1, 3, H, W] and normalize
        let mut pixel_values = Array4::<f32>::zeros((1, 3, self.image_size as usize, self.image_size as usize));

        for y in 0..self.image_size as usize {
            for x in 0..self.image_size as usize {
                let pixel = resized.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    // Normalize: (pixel - mean) / std
                    pixel_values[[0, c, y, x]] = (pixel[c] as f32 - self.mean[c]) / self.std[c];
                }
            }
        }

        Ok((pixel_values, original_size, resized_size))
    }

    /// Transform point coordinates from original image space to resized image space
    /// Points are in (x, y) format
    pub fn transform_points(&self, points: &[(f32, f32)], original_size: (u32, u32)) -> Vec<(f32, f32)> {
        let (orig_w, orig_h) = original_size;
        let scale_x = self.image_size as f32 / orig_w as f32;
        let scale_y = self.image_size as f32 / orig_h as f32;

        points.iter()
            .map(|(x, y)| (x * scale_x, y * scale_y))
            .collect()
    }

    /// Transform box coordinates from original image space to resized image space
    /// Boxes are in (x1, y1, x2, y2) format
    pub fn transform_boxes(&self, boxes: &[(f32, f32, f32, f32)], original_size: (u32, u32)) -> Vec<(f32, f32, f32, f32)> {
        let (orig_w, orig_h) = original_size;
        let scale_x = self.image_size as f32 / orig_w as f32;
        let scale_y = self.image_size as f32 / orig_h as f32;

        boxes.iter()
            .map(|(x1, y1, x2, y2)| {
                (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
            })
            .collect()
    }

    /// Resize mask from model output size to original image size
    /// Returns a grayscale image with binary mask (0 or 255)
    pub fn postprocess_mask(
        &self,
        mask: &ndarray::ArrayView2<f32>,
        original_size: (u32, u32),
    ) -> GrayImage {
        let (orig_w, orig_h) = original_size;
        let (mask_h, mask_w) = (mask.shape()[0], mask.shape()[1]);

        // Create a grayscale image from the mask
        let mut mask_image = GrayImage::new(mask_w as u32, mask_h as u32);
        for y in 0..mask_h {
            for x in 0..mask_w {
                // Apply sigmoid and threshold at 0.5
                let val = mask[[y, x]];
                let sigmoid = 1.0 / (1.0 + (-val).exp());
                let pixel_val = if sigmoid > 0.5 { 255u8 } else { 0u8 };
                mask_image.put_pixel(x as u32, y as u32, image::Luma([pixel_val]));
            }
        }

        // Resize to original image size
        image::imageops::resize(
            &mask_image,
            orig_w,
            orig_h,
            FilterType::Lanczos3,
        )
    }

    /// Convert a grayscale mask to RGBA with a specified color
    /// The mask alpha is based on the mask intensity
    pub fn mask_to_rgba(&self, mask: &GrayImage, color: [u8; 3], opacity: u8) -> RgbaImage {
        let (w, h) = mask.dimensions();
        let mut rgba = RgbaImage::new(w, h);

        for y in 0..h {
            for x in 0..w {
                let mask_val = mask.get_pixel(x, y).0[0];
                if mask_val > 127 {
                    // Foreground pixel - use the specified color with opacity
                    rgba.put_pixel(x, y, image::Rgba([color[0], color[1], color[2], opacity]));
                } else {
                    // Background pixel - transparent
                    rgba.put_pixel(x, y, image::Rgba([0, 0, 0, 0]));
                }
            }
        }

        rgba
    }

    /// Encode a grayscale mask as base64 PNG
    pub fn mask_to_base64_png(&self, mask: &GrayImage) -> Result<String> {
        let mut buffer = Cursor::new(Vec::new());
        mask.write_to(&mut buffer, image::ImageFormat::Png)?;
        Ok(BASE64.encode(buffer.into_inner()))
    }

    /// Encode an RGBA image as base64 PNG
    pub fn rgba_to_base64_png(&self, image: &RgbaImage) -> Result<String> {
        let mut buffer = Cursor::new(Vec::new());
        image.write_to(&mut buffer, image::ImageFormat::Png)?;
        Ok(BASE64.encode(buffer.into_inner()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_processor_creation() {
        let processor = Sam3ImageProcessor::new();
        assert_eq!(processor.image_size, 1008);
    }

    #[test]
    fn test_point_transformation() {
        let processor = Sam3ImageProcessor::new();
        let original_size = (2016, 2016); // 2x the target size
        let points = vec![(100.0, 200.0)];
        let transformed = processor.transform_points(&points, original_size);

        // Points should be scaled by 0.5
        assert!((transformed[0].0 - 50.0).abs() < 0.01);
        assert!((transformed[0].1 - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_box_transformation() {
        let processor = Sam3ImageProcessor::new();
        let original_size = (2016, 2016);
        let boxes = vec![(100.0, 100.0, 200.0, 200.0)];
        let transformed = processor.transform_boxes(&boxes, original_size);

        assert!((transformed[0].0 - 50.0).abs() < 0.01);
        assert!((transformed[0].1 - 50.0).abs() < 0.01);
        assert!((transformed[0].2 - 100.0).abs() < 0.01);
        assert!((transformed[0].3 - 100.0).abs() < 0.01);
    }
}
