#![feature(portable_simd)]

pub(crate) mod batch;
#[cfg(feature = "python")]
pub(crate) mod bindings;
pub(crate) mod bpe;
#[cfg(feature = "python")]
pub(crate) mod bpe_train;
pub(crate) mod input;
pub mod load_tokenizer;
pub mod pretokenize;
#[cfg(test)]
pub(crate) mod test_hub;
pub(crate) mod token;

#[cfg(feature = "python")]
mod python;

pub use crate::batch::{WorkerPool, encode_docs_ragged, sp_encode_docs_ragged};
pub use crate::bpe::Tokenizer;
pub use crate::bpe::sentencepiece::EncodeState;
