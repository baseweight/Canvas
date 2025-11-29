// Audio decoding utilities for Baseweight Canvas
// Decodes WAV, MP3, FLAC audio files to PCM F32 format for mtmd

use anyhow::{Result, anyhow};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use std::fs::File;
use std::path::Path;

/// Decoded audio data in PCM F32 format
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u32,
}

/// Resample audio to a target sample rate using linear interpolation
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (samples.len() as f64 / ratio) as usize;
    let mut resampled = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let pos = i as f64 * ratio;
        let index = pos as usize;
        let frac = pos - index as f64;

        if index + 1 < samples.len() {
            // Linear interpolation
            let sample = samples[index] * (1.0 - frac) as f32 + samples[index + 1] * frac as f32;
            resampled.push(sample);
        } else if index < samples.len() {
            resampled.push(samples[index]);
        }
    }

    resampled
}

/// Decode an audio file to PCM F32 mono format
/// Audio will be converted to mono if it has multiple channels
pub fn decode_audio_file(path: &str) -> Result<AudioData> {
    let path_obj = Path::new(path);

    // Open the audio file
    let file = File::open(path_obj)?;

    // Create the media source stream
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Create a hint to help the format registry guess the format
    let mut hint = Hint::new();
    if let Some(extension) = path_obj.extension() {
        if let Some(ext_str) = extension.to_str() {
            hint.with_extension(ext_str);
        }
    }

    // Use the default options for metadata and format readers
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    // Probe the media source
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)?;

    // Get the instantiated format reader
    let mut format = probed.format;

    // Find the first audio track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("No audio tracks found"))?;

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate
        .ok_or_else(|| anyhow!("Sample rate not found"))?;
    let channels = track.codec_params.channels
        .ok_or_else(|| anyhow!("Channel info not found"))?
        .count() as u32;

    println!("Audio file info: {} Hz, {} channels", sample_rate, channels);

    // Create a decoder for the track
    let dec_opts: DecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)?;

    // Decode all audio packets
    let mut samples = Vec::new();

    loop {
        // Get the next packet from the format reader
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(err) => return Err(err.into()),
        };

        // Only decode packets for the selected track
        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet into audio samples
        match decoder.decode(&packet)? {
            AudioBufferRef::F32(buf) => {
                // Convert to mono by averaging channels
                for i in 0..buf.frames() {
                    let mut sum = 0.0;
                    for ch in 0..buf.spec().channels.count() {
                        sum += buf.chan(ch)[i];
                    }
                    samples.push(sum / buf.spec().channels.count() as f32);
                }
            }
            AudioBufferRef::U8(buf) => {
                for i in 0..buf.frames() {
                    let mut sum = 0.0;
                    for ch in 0..buf.spec().channels.count() {
                        // Convert U8 to F32: (sample / 128.0) - 1.0
                        let sample = (buf.chan(ch)[i] as f32 / 128.0) - 1.0;
                        sum += sample;
                    }
                    samples.push(sum / buf.spec().channels.count() as f32);
                }
            }
            AudioBufferRef::U16(buf) => {
                for i in 0..buf.frames() {
                    let mut sum = 0.0;
                    for ch in 0..buf.spec().channels.count() {
                        // Convert U16 to F32: (sample / 32768.0) - 1.0
                        let sample = (buf.chan(ch)[i] as f32 / 32768.0) - 1.0;
                        sum += sample;
                    }
                    samples.push(sum / buf.spec().channels.count() as f32);
                }
            }
            AudioBufferRef::U24(buf) => {
                for i in 0..buf.frames() {
                    let mut sum = 0.0;
                    for ch in 0..buf.spec().channels.count() {
                        // Convert U24 to F32
                        let sample = (buf.chan(ch)[i].into_u32() as f32 / 8388608.0) - 1.0;
                        sum += sample;
                    }
                    samples.push(sum / buf.spec().channels.count() as f32);
                }
            }
            AudioBufferRef::U32(buf) => {
                for i in 0..buf.frames() {
                    let mut sum = 0.0;
                    for ch in 0..buf.spec().channels.count() {
                        // Convert U32 to F32
                        let sample = (buf.chan(ch)[i] as f32 / 2147483648.0) - 1.0;
                        sum += sample;
                    }
                    samples.push(sum / buf.spec().channels.count() as f32);
                }
            }
            AudioBufferRef::S8(buf) => {
                for i in 0..buf.frames() {
                    let mut sum = 0.0;
                    for ch in 0..buf.spec().channels.count() {
                        // Convert S8 to F32
                        let sample = buf.chan(ch)[i] as f32 / 128.0;
                        sum += sample;
                    }
                    samples.push(sum / buf.spec().channels.count() as f32);
                }
            }
            AudioBufferRef::S16(buf) => {
                for i in 0..buf.frames() {
                    let mut sum = 0.0;
                    for ch in 0..buf.spec().channels.count() {
                        // Convert S16 to F32
                        let sample = buf.chan(ch)[i] as f32 / 32768.0;
                        sum += sample;
                    }
                    samples.push(sum / buf.spec().channels.count() as f32);
                }
            }
            AudioBufferRef::S24(buf) => {
                for i in 0..buf.frames() {
                    let mut sum = 0.0;
                    for ch in 0..buf.spec().channels.count() {
                        // Convert S24 to F32
                        let sample = buf.chan(ch)[i].inner() as f32 / 8388608.0;
                        sum += sample;
                    }
                    samples.push(sum / buf.spec().channels.count() as f32);
                }
            }
            AudioBufferRef::S32(buf) => {
                for i in 0..buf.frames() {
                    let mut sum = 0.0;
                    for ch in 0..buf.spec().channels.count() {
                        // Convert S32 to F32
                        let sample = buf.chan(ch)[i] as f32 / 2147483648.0;
                        sum += sample;
                    }
                    samples.push(sum / buf.spec().channels.count() as f32);
                }
            }
            AudioBufferRef::F64(buf) => {
                for i in 0..buf.frames() {
                    let mut sum = 0.0;
                    for ch in 0..buf.spec().channels.count() {
                        sum += buf.chan(ch)[i] as f32;
                    }
                    samples.push(sum / buf.spec().channels.count() as f32);
                }
            }
        }
    }

    println!("Decoded {} samples", samples.len());

    Ok(AudioData {
        samples,
        sample_rate,
        channels,
    })
}
