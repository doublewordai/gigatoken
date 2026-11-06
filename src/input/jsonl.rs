use std::io::BufRead;

use crate::input::Document;
#[cfg(not(target_arch = "x86"))]
use sonic_rs::JsonValueTrait;

#[cfg(not(target_arch = "x86"))]
fn text_document(line: &[u8], field: &str) -> Option<Document<'static>> {
    let value = sonic_rs::get_from_slice(line, &[field]).ok()?;
    Some(Document::from(value.as_str()?.as_bytes().to_vec()))
}

#[cfg(target_arch = "x86")]
use portable_text_document as text_document;

#[cfg(any(target_arch = "x86", test))]
fn portable_text_document(line: &[u8], field: &str) -> Option<Document<'static>> {
    let parsed: serde_json::Value = serde_json::from_slice(line).ok()?;
    Some(Document::from(
        parsed.get(field)?.as_str()?.as_bytes().to_vec(),
    ))
}

/// Zero-copy JSONL iterator over a byte slice (e.g. from mmap).
/// Used for parallel chunking of uncompressed JSONL files.
pub struct JsonLinesSlice<'a> {
    slice: &'a [u8],
    position: usize,
    field: &'a str,
}

impl<'a> JsonLinesSlice<'a> {
    pub fn new(slice: &'a [u8], field: &'a str) -> Self {
        Self {
            slice,
            position: 0,
            field,
        }
    }
}

impl<'a> Iterator for JsonLinesSlice<'a> {
    type Item = Document<'static>; // Owned: JSON parsing requires extraction

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Skip newlines
            while self.position < self.slice.len() && self.slice[self.position] == b'\n' {
                self.position += 1;
            }
            if self.position >= self.slice.len() {
                return None;
            }

            let line_end = memchr::memchr(b'\n', &self.slice[self.position..])
                .map(|i| self.position + i)
                .unwrap_or(self.slice.len());
            let line = &self.slice[self.position..line_end];
            self.position = line_end + 1;

            if line.is_empty() {
                continue;
            }

            return text_document(line, self.field);
        }
    }
}

/// Streaming JSONL iterator over a `BufRead` source.
/// Reads one line at a time — never buffers the entire file.
pub(crate) struct JsonLinesReader<R> {
    reader: R,
    field: String,
    line_buf: Vec<u8>,
}

impl<R: BufRead> JsonLinesReader<R> {
    pub(crate) fn new(reader: R, field: &str) -> Self {
        Self {
            reader,
            field: field.to_string(),
            line_buf: Vec::with_capacity(4096),
        }
    }
}

impl<R: BufRead> Iterator for JsonLinesReader<R> {
    type Item = Document<'static>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.line_buf.clear();
            let bytes_read = self.reader.read_until(b'\n', &mut self.line_buf).ok()?;
            if bytes_read == 0 {
                return None; // EOF
            }

            let line = self.line_buf.as_slice();
            // Skip empty lines
            if line.iter().all(|&b| b == b'\n' || b == b'\r') {
                continue;
            }

            return text_document(line, &self.field);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readers_decode_escaped_strings_and_unicode() {
        let data = b"{\"text\":\"hello\\n\\u03bb\"}\n{\"text\":\"next\"}";
        let sliced: Vec<_> = JsonLinesSlice::new(data, "text")
            .map(|d| d.as_ref().to_vec())
            .collect();
        let streamed: Vec<_> = JsonLinesReader::new(&data[..], "text")
            .map(|d| d.as_ref().to_vec())
            .collect();
        assert_eq!(
            sliced,
            vec!["hello\nλ".as_bytes().to_vec(), b"next".to_vec()]
        );
        assert_eq!(streamed, sliced);
    }

    #[test]
    fn portable_parser_matches_simd_for_text_fields() {
        for line in [
            br#"{"text":"hello\n\u03bb"}"#.as_slice(),
            br#"{"text":"next"}"#,
            br#"{"other":"value"}"#,
            br#"{"text":42}"#,
            b"invalid",
        ] {
            assert_eq!(
                portable_text_document(line, "text").map(|d| d.as_ref().to_vec()),
                text_document(line, "text").map(|d| d.as_ref().to_vec()),
            );
        }
    }

    #[test]
    fn rejects_missing_and_non_string_fields() {
        assert!(text_document(br#"{"other":"value"}"#, "text").is_none());
        assert!(text_document(br#"{"text":42}"#, "text").is_none());
        assert!(text_document(b"invalid", "text").is_none());
    }
}
